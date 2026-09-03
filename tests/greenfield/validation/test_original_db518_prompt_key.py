from __future__ import annotations

from hashlib import sha256

import pytest

from glm_tpu.greenfield.validation import original_db518_prompt_key as db518


def _optimized_hlo() -> str:
    entry_parameters = (
        "cache: bf16[24,16,32,128], table: s32[16], "
        "embeddings: bf16[37,6144], rows: s32[2048], "
        "positions: s32[2048], norm: bf16[6144], "
        "wk: f32[128,6144], weight: bf16[128], bias: bf16[128]")
    return ("HloModule test, " + db518._ENTRY_LAYOUT + "\n\n" +
            "FileNames\n1 \"different/callsite.py\"\n\n" +
            "FunctionNames\n1 \"main\"\n\n" +
            "FileLocations\n1 {file_name_id=1 function_name_id=1 line=9}\n\n" +
            "StackFrames\n1 {file_location_id=1 parent_frame_id=1}\n\n" +
            "%copy (arg: bf16[1]) -> bf16[1] {\n" +
            "  %arg = bf16[1]{0} parameter(0)\n" +
            "  ROOT %same = bf16[1]{0} copy(%arg), " +
            "metadata={op_name=\"changing/source\"}\n}\n\n" +
            f"ENTRY %main ({entry_parameters}) -> bf16[24,16,32,128] {{\n" +
            "  %cache = bf16[24,16,32,128]{3,2,1,0} parameter(0)\n" +
            "  ROOT %result = bf16[24,16,32,128]{3,2,1,0} copy(%cache), " +
            "metadata={op_name=\"changing/root\"}, " +
            "frontend_attributes={xla.sdy.sharding=\"ignored\"}\n}\n")


def _admit_synthetic(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(
        db518,
        "validate_prompt_index_key_association_hlo",
        lambda *_args, **_kwargs: {
            "passed": True,
            "violations": []
        },
    )
    canonical = db518.canonicalize_original_db518_hlo(text)
    monkeypatch.setattr(
        db518,
        "ORIGINAL_DB518_CANONICAL_HLO_SHA256",
        sha256(canonical.encode()).hexdigest(),
    )


def test_exact_wrapper_accepts_source_metadata_only_drift(monkeypatch):
    text = _optimized_hlo()
    _admit_synthetic(monkeypatch, text)
    changed = text.replace("changing/source",
                           "new/source").replace("changing/root", "new/root")
    result = db518.validate_original_db518_prompt_key_hlo(changed)
    assert result["passed"] is True
    assert result["violations"] == []


def test_exact_wrapper_rejects_graph_drift(monkeypatch):
    text = _optimized_hlo()
    _admit_synthetic(monkeypatch, text)
    changed = text.replace("copy(%cache)", "bitcast(%cache)")
    result = db518.validate_original_db518_prompt_key_hlo(changed)
    assert result["passed"] is False
    assert "DB518 canonical optimized-HLO graph drifted" in result[
        "violations"]


def test_exact_wrapper_rejects_layout_or_root_drift(monkeypatch):
    text = _optimized_hlo()
    _admit_synthetic(monkeypatch, text)
    layout = text.replace("bf16[37,6144]", "bf16[38,6144]", 1)
    assert not db518.validate_original_db518_prompt_key_hlo(layout)["passed"]
    root = text.replace("ROOT %result", "%result")
    assert not db518.validate_original_db518_prompt_key_hlo(root)["passed"]


def test_canonicalizer_rejects_malformed_fields():
    text = _optimized_hlo().replace(
        'metadata={op_name="changing/root"}',
        'metadata={op_name="changing/root"',
    )
    with pytest.raises(ValueError, match="malformed optimized-HLO field"):
        db518.canonicalize_original_db518_hlo(text)


def _stablehlo() -> str:
    return "\n".join((
        "module @jit_layer0_prompt_index_key_gather_cache_chunk {",
        "  // tensor<24x16x32x128xbf16>",
        "  // tensor<16xi32>",
        "  // tensor<37x6144xbf16>",
        "  // tensor<2048xi32>",
        "  // tensor<6144xbf16>",
        "  // tensor<128x6144xf32>",
        "  // tensor<128xbf16>",
        "  %0 = stablehlo.while ...",
        ('  %1 = "stablehlo.scatter"(%arg0, %arg1, %arg2) '
         '<{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims '
         '= [1]>}> ...'),
        "  %2 = stablehlo.cosine ...",
        "  %3 = stablehlo.sine ...",
        "}",
    ))


def test_stablehlo_contract_accepts_exact_surface_and_rejects_bypasses():
    text = _stablehlo()
    assert text.count("stablehlo.scatter") == 2
    assert db518.validate_original_db518_prompt_key_stablehlo(text)["passed"]
    assert not db518.validate_original_db518_prompt_key_stablehlo(
        text + "\nstablehlo.all_reduce")["passed"]
    assert not db518.validate_original_db518_prompt_key_stablehlo(
        text.replace('"stablehlo.scatter"', '"stablehlo.add"'))["passed"]
    assert not db518.validate_original_db518_prompt_key_stablehlo(
        text + '\n  %9 = "stablehlo.scatter"(%a, %b, %c) ...')["passed"]
