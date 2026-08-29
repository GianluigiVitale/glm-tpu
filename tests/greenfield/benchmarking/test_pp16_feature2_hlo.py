from __future__ import annotations

import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

import glm_tpu.greenfield.benchmarking.pp16_feature2_hlo as feature2_hlo_module
from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import (
    canonicalize_feature2_optimized_hlo,
    validate_feature2_main_optimized_hlo,
    validate_feature2_main_stablehlo,
    validate_feature2_materializer_optimized_hlo,
    validate_feature2_sealed_hlo_archive_identity,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError

REAL_ACQUIRED_MAIN_HLO = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_prefill_acquire_"
    "20260828T231028891602866Z/hlo/feature2_main.optimized_hlo.txt"
)
REAL_ACQUIRED_MAIN_STABLEHLO = (
    REAL_ACQUIRED_MAIN_HLO.parent / "feature2_main.stablehlo.mlir"
)
REAL_NUMERICAL_REFUSAL_MAIN_HLO = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_prefill_numerical_"
    "20260828T234600895206573Z/hlo/feature2_main.optimized_hlo.txt"
)
EXPECTED_CANONICAL_SHA256 = (
    "fb5aaf025005f3fbb5a3c66e6a719ec3a78fb86d344afcf6288e6e93720310f7"
)
REAL_SEALED_RUN = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_prefill_acquire_"
    "20260829T034414037011891Z/hlo"
)
REAL_SEALED_STABLEHLO = REAL_SEALED_RUN / "feature2_main.stablehlo.mlir"
REAL_SEALED_OPTIMIZED_HLO = REAL_SEALED_RUN / "feature2_main.optimized_hlo.txt"
REAL_POSITION113_RUN = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_position113_acquire_"
    "20260829T072101902362381Z/hlo"
)
REAL_POSITION113_STABLEHLO = REAL_POSITION113_RUN / "feature2_main.stablehlo.mlir"
REAL_POSITION113_OPTIMIZED_HLO = (
    REAL_POSITION113_RUN / "feature2_main.optimized_hlo.txt"
)
REAL_POSITION113_CANONICAL_HLO = (
    REAL_POSITION113_RUN / "feature2_main.execution_canonical_hlo.txt"
)
REAL_DB518_RUN = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_acquire_"
    "20260829T101159660007493Z/hlo"
)
REAL_DB518_STABLEHLO = REAL_DB518_RUN / "feature2_main.stablehlo.mlir"
REAL_DB518_OPTIMIZED_HLO = REAL_DB518_RUN / "feature2_main.optimized_hlo.txt"
REAL_DB518_CANONICAL_HLO = REAL_DB518_RUN / "feature2_main.execution_canonical_hlo.txt"


def _pin_synthetic_sealed_optimized_hlo(
    monkeypatch: pytest.MonkeyPatch,
    optimized_hlo: str,
) -> None:
    """Give a synthetic fixture the same immutable-identity behavior as TPU HLO."""

    expected_sha = sha256(optimized_hlo.encode()).hexdigest()
    expected_bytes = len(optimized_hlo.encode())
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_SEALED_OPTIMIZED_CANONICAL_SHA256",
        expected_sha,
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_SEALED_OPTIMIZED_CANONICAL_BYTES",
        expected_bytes,
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_SEALED_OPTIMIZED_STACK_FRAME_REFERENCES",
        0,
    )

    def synthetic_canonicalizer(value: str) -> tuple[str, dict[str, object]]:
        return value, {
            "byte_count": len(value.encode()),
            "canonicalizer_version": 1,
            "debug_sections": {},
            "sha256": sha256(value.encode()).hexdigest(),
            "stripped_stack_frame_references": 0,
        }

    monkeypatch.setattr(
        feature2_hlo_module,
        "canonicalize_feature2_optimized_hlo",
        synthetic_canonicalizer,
    )


def _pin_synthetic_position113_optimized_hlo(
    monkeypatch: pytest.MonkeyPatch,
    optimized_hlo: str,
) -> None:
    _pin_synthetic_sealed_optimized_hlo(monkeypatch, optimized_hlo)
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_CANONICAL_SHA256",
        sha256(optimized_hlo.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_CANONICAL_BYTES",
        len(optimized_hlo.encode()),
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_STACK_FRAME_REFERENCES",
        0,
    )


def _pin_synthetic_sealed_stablehlo(
    monkeypatch: pytest.MonkeyPatch,
    stablehlo: str,
) -> None:
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_SEALED_STABLEHLO_SHA256",
        sha256(stablehlo.encode()).hexdigest(),
    )


def _pin_synthetic_position113_stablehlo(
    monkeypatch: pytest.MonkeyPatch,
    stablehlo: str,
) -> None:
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_STABLEHLO_SHA256",
        sha256(stablehlo.encode()).hexdigest(),
    )


def _optimized_hlo_with_debug_provenance(
    *, frame: int = 7, extra_debug_rows: bool = False
) -> str:
    extra_file = '2 "wrapper.py"\n' if extra_debug_rows else ""
    extra_function = '2 "wrapper"\n' if extra_debug_rows else ""
    extra_location = (
        "2 {file_name_id=2 function_name_id=2 line=2 end_line=2 "
        "column=1 end_column=1}\n"
        if extra_debug_rows
        else ""
    )
    extra_frame = (
        "2 {file_location_id=2 parent_frame_id=2}\n" if extra_debug_rows else ""
    )
    return (
        "HloModule canonical_test, num_partitions=2\n\n"
        "FileNames\n"
        '1 "program.py"\n'
        f"{extra_file}\n"
        "FunctionNames\n"
        '1 "execute"\n'
        f"{extra_function}\n"
        "FileLocations\n"
        "1 {file_name_id=1 function_name_id=1 line=1 end_line=1 "
        "column=1 end_column=1}\n"
        f"{extra_location}\n"
        "StackFrames\n"
        "1 {file_location_id=1 parent_frame_id=1}\n"
        f"{extra_frame}\n\n"
        "%fused (x: s32[]) -> s32[] {\n"
        "  %x = s32[] parameter(0), "
        f'metadata={{op_name="execute/add" stack_frame_id={frame}}}\n'
        "  ROOT %root = s32[] add(%x, %x), "
        f'metadata={{op_name="execute/add" stack_frame_id={frame}}}\n'
        "}\n"
    )


def test_feature2_canonical_hlo_removes_only_debug_provenance() -> None:
    first, first_report = canonicalize_feature2_optimized_hlo(
        _optimized_hlo_with_debug_provenance()
    )
    second, second_report = canonicalize_feature2_optimized_hlo(
        _optimized_hlo_with_debug_provenance(frame=99, extra_debug_rows=True)
    )
    assert first == second
    assert first_report["sha256"] == second_report["sha256"]
    assert first_report["stripped_stack_frame_references"] == 2
    assert second_report["stripped_stack_frame_references"] == 2
    assert 'op_name="execute/add"' in first
    assert "stack_frame_id=" not in first


@pytest.mark.parametrize(
    "old,new",
    (
        ("s32[] add(%x, %x)", "s32[] multiply(%x, %x)"),
        ("s32[] parameter(0)", "s32[] constant(0)"),
        ('op_name="execute/add"', 'op_name="execute/mul"'),
        ('op_name="execute/add"', 'op_name="stack_frame_id=7"'),
        ('op_name="execute/add"', 'op_name="execute/add" future_key=1'),
        ("ROOT %root", "  ROOT %renamed"),
        ("%x, %x", "%x, %root"),
        ("HloModule canonical_test", "HloModule  canonical_test"),
    ),
)
def test_feature2_canonical_hlo_preserves_computational_and_unknown_text(
    old: str, new: str
) -> None:
    baseline, baseline_report = canonicalize_feature2_optimized_hlo(
        _optimized_hlo_with_debug_provenance()
    )
    mutated, mutated_report = canonicalize_feature2_optimized_hlo(
        _optimized_hlo_with_debug_provenance().replace(old, new, 1)
    )
    assert mutated != baseline
    assert mutated_report["sha256"] != baseline_report["sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("FileNames\n", "", 1),
        lambda value: value.replace("FileNames\n", "FileNames\nFileNames\n", 1),
        lambda value: value.replace("\nFileNames\n", "\nrelocated\nFileNames\n", 1),
        lambda value: value.replace('1 "program.py"', "1 malformed"),
        lambda value: value.replace("stack_frame_id=7}", "stack_frame_id=x}"),
    ),
)
def test_feature2_canonical_hlo_refuses_malformed_debug_provenance(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        canonicalize_feature2_optimized_hlo(
            mutation(_optimized_hlo_with_debug_provenance())
        )


def test_feature2_canonical_hlo_refuses_stack_frame_outside_metadata() -> None:
    original = _optimized_hlo_with_debug_provenance()
    attacked = original.replace(" stack_frame_id=7}", "", 1).replace(
        ", metadata={",
        ", frontend_attributes={stack_frame_id=999999}, metadata={",
        1,
    )
    with pytest.raises(BenchmarkValidationError):
        canonicalize_feature2_optimized_hlo(attacked)


@pytest.mark.skipif(
    not REAL_ACQUIRED_MAIN_HLO.is_file()
    or not REAL_NUMERICAL_REFUSAL_MAIN_HLO.is_file(),
    reason="protected PP16 HLO pair is unavailable",
)
def test_feature2_real_hlo_pair_has_one_execution_canonical_identity() -> None:
    acquired_raw = REAL_ACQUIRED_MAIN_HLO.read_text()
    refused_raw = REAL_NUMERICAL_REFUSAL_MAIN_HLO.read_text()
    assert (
        sha256(acquired_raw.encode()).hexdigest()
        != sha256(refused_raw.encode()).hexdigest()
    )
    acquired, acquired_report = canonicalize_feature2_optimized_hlo(acquired_raw)
    refused, refused_report = canonicalize_feature2_optimized_hlo(refused_raw)
    assert acquired == refused
    assert acquired_report["sha256"] == EXPECTED_CANONICAL_SHA256
    assert refused_report["sha256"] == EXPECTED_CANONICAL_SHA256
    assert acquired_report["byte_count"] == 7_870_521
    assert refused_report["byte_count"] == 7_870_521
    assert acquired_report["stripped_stack_frame_references"] == 16_170
    assert refused_report["stripped_stack_frame_references"] == 16_170


@pytest.mark.skipif(
    not REAL_SEALED_STABLEHLO.is_file() or not REAL_SEALED_OPTIMIZED_HLO.is_file(),
    reason="protected sealed PP16 HLO is unavailable",
)
def test_feature2_real_sealed_hlo_authenticates_complete_producer_identity() -> None:
    stable = validate_feature2_main_stablehlo(
        REAL_SEALED_STABLEHLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    optimized = validate_feature2_main_optimized_hlo(
        REAL_SEALED_OPTIMIZED_HLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    assert stable["stablehlo_sha256"] == (
        "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4"
    )
    assert optimized["sealed_canonical_hlo_identity"]["sha256"] == (
        "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
    )
    assert optimized["sealed_canonical_hlo_identity"]["byte_count"] == 6_558_627
    assert (
        optimized["sealed_canonical_hlo_identity"]["stripped_stack_frame_references"]
        == 14_561
    )


@pytest.mark.skipif(
    not REAL_POSITION113_STABLEHLO.is_file()
    or not REAL_POSITION113_OPTIMIZED_HLO.is_file()
    or not REAL_POSITION113_CANONICAL_HLO.is_file(),
    reason="protected position-113 PP16 HLO is unavailable",
)
def test_feature2_real_position113_hlo_authenticates_complete_producer_identity() -> (
    None
):
    stable = validate_feature2_main_stablehlo(
        REAL_POSITION113_STABLEHLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
        source_jaxpr_sha256=(
            "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
        ),
    )
    optimized = validate_feature2_main_optimized_hlo(
        REAL_POSITION113_OPTIMIZED_HLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    archive = validate_feature2_sealed_hlo_archive_identity(
        REAL_POSITION113_STABLEHLO,
        REAL_POSITION113_OPTIMIZED_HLO,
        REAL_POSITION113_CANONICAL_HLO,
        expected_stablehlo_sha256=(
            "bc2fcc77e84217ee0264e3856309f298e61e70ac0f4e3ecebecf72d290c26035"
        ),
        expected_canonical_sha256=(
            "5b5dfacf015c579f41661868486e627bb383322fe8fb5c64597c259e85c4a10e"
        ),
        expected_canonical_bytes=6_662_190,
        expected_canonicalizer_version=1,
        expected_stripped_stack_frame_references=14_781,
    )
    assert stable["stablehlo_sha256"] == archive["stablehlo_sha256"]
    assert (
        optimized["sealed_canonical_hlo_identity"]["sha256"]
        == (archive["canonical_hlo_identity"]["sha256"])
    )
    assert optimized["sealed_bindings"] == {
        str(index): marker
        for index, marker in feature2_hlo_module._SEALED_BOUNDARY_ROOT_MARKERS.items()
    }
    assert optimized["position113_observer_bindings"] == {
        str(index): marker
        for index, marker in feature2_hlo_module._POSITION113_OBSERVER_ROOT_MARKERS.items()
    }
    assert optimized["position113_observer_root_hints_causal"] is True
    assert optimized["position113_lineage"]["scan_count"] == 4
    assert optimized["position113_lineage"]["scan_handoff_field_count"] == 27
    assert optimized["position113_lineage"]["branch_topology_record_count"] == 32
    assert optimized["position113_lineage"]["branch_topology_sha256"] == (
        "05322ce7ad09db5463e17d4a7e31142edaffe128abe5e9d85d5ee81f5e3eb02b"
    )
    assert [
        scan["causal_observer_field_count"]
        for scan in optimized["position113_lineage"]["scans"]
    ] == [9, 9, 9, 9]


@pytest.mark.skipif(
    not REAL_DB518_STABLEHLO.is_file()
    or not REAL_DB518_OPTIMIZED_HLO.is_file()
    or not REAL_DB518_CANONICAL_HLO.is_file(),
    reason="protected DB518 PP16 HLO is unavailable",
)
def test_feature2_real_db518_hlo_authenticates_causal_key_consumption() -> None:
    stable = validate_feature2_main_stablehlo(
        REAL_DB518_STABLEHLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
        exact_layer0_prompt_keys=True,
        source_jaxpr_sha256=(
            "f87c0f162523bf6221d2824aa4b7268f1c770a55d2f8cde6da7506f595e9447f"
        ),
    )
    optimized = validate_feature2_main_optimized_hlo(
        REAL_DB518_OPTIMIZED_HLO.read_text(),
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
        exact_layer0_prompt_keys=True,
    )
    archive = validate_feature2_sealed_hlo_archive_identity(
        REAL_DB518_STABLEHLO,
        REAL_DB518_OPTIMIZED_HLO,
        REAL_DB518_CANONICAL_HLO,
        expected_stablehlo_sha256=(
            "a79d4823db0fecf8b1bc980b16df283bbe7a497795fa06ff783d3cde8fa98f6b"
        ),
        expected_canonical_sha256=(
            "56b9b88dc081dd5d3ea2219d46be641128cbed6e9759754be7f87578b51b5d8c"
        ),
        expected_canonical_bytes=6_863_602,
        expected_canonicalizer_version=1,
        expected_stripped_stack_frame_references=15_339,
    )
    assert stable["stablehlo_sha256"] == archive["stablehlo_sha256"]
    assert stable["identity_acquisition_only"] is False
    assert optimized["identity_acquisition_only"] is False
    assert (
        optimized["sealed_canonical_hlo_identity"]["sha256"]
        == (archive["canonical_hlo_identity"]["sha256"])
    )
    assert optimized["position113_lineage"]["causal"] is True
    assert optimized["position113_lineage"]["branch_topology_sha256"] == (
        "92a1b18b523984dd98d2579d25668b6bd582f6f9f23a26d9e4449b29a81ee46d"
    )
    key_lineage = optimized["db518_layer0_key_lineage"]
    assert key_lineage["causal"] is True
    assert key_lineage["scan_count"] == 4
    assert key_lineage["cache_write_count"] == 4
    assert key_lineage["physical_m64_projection_count"] == 16
    assert key_lineage["topology_sha256"] == (
        "235dfa726f82a152bf0218506cf8d3345520a7673ab7a6b908ef06df821465bd"
    )
    assert [scan["cache_seed_kind"] for scan in key_lineage["scans"]] == [
        "zero",
        "handoff",
        "handoff",
        "handoff",
    ]


def _crosswire_db518_first_key_chunk(value: str) -> str:
    original = "%pad_maximum_fusion.36, /*index=30*/"
    assert value.count(original) == 1
    return value.replace(original, "%pad_maximum_fusion.37, /*index=30*/", 1)


def _disconnect_db518_first_cache_write(value: str) -> str:
    original = "fusion(%conditional.35, %pad_clamp_fusion.19)"
    assert value.count(original) == 1
    return value.replace(original, "fusion(%copy-done.31, %pad_clamp_fusion.19)", 1)


@pytest.mark.skipif(
    not REAL_DB518_OPTIMIZED_HLO.is_file(),
    reason="protected DB518 PP16 HLO is unavailable",
)
@pytest.mark.parametrize(
    "mutation",
    (
        _crosswire_db518_first_key_chunk,
        lambda value: value.replace(
            "tuple(%copy.991, %get-tuple-element.18251, %get-tuple-element.18252,",
            "tuple(%copy.991, %get-tuple-element.18251, %broadcast_in_dim.3762,",
            1,
        ),
        lambda value: value.replace(
            "fusion(%get-tuple-element.17095, %get-tuple-element.16978)",
            "fusion(%get-tuple-element.17095, %get-tuple-element.16979)",
            1,
        ),
        lambda value: value.replace(
            "tuple(%bitcast.6336, %select_n.5970, %fusion.3617, %copy-done.31)",
            "tuple(%bitcast.6336, %select_n.5970, %fusion.3618, %copy-done.31)",
            1,
        ),
        _disconnect_db518_first_cache_write,
        lambda value: value.replace("constant(113)", "constant(114)", 1),
    ),
)
def test_feature2_db518_lineage_refuses_mutation_even_with_rebased_identity(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    original = REAL_DB518_OPTIMIZED_HLO.read_text()
    attacked = mutation(original)
    assert attacked != original
    _, identity = canonicalize_feature2_optimized_hlo(attacked)
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_DB518_OPTIMIZED_CANONICAL_SHA256",
        identity["sha256"],
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_DB518_OPTIMIZED_CANONICAL_BYTES",
        identity["byte_count"],
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_DB518_OPTIMIZED_STACK_FRAME_REFERENCES",
        identity["stripped_stack_frame_references"],
    )
    with pytest.raises(BenchmarkValidationError, match="DB518|position-113"):
        validate_feature2_main_optimized_hlo(
            attacked,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            exact_layer0_prompt_keys=True,
        )


def _swap_position113_current_key_padding_ranges(value: str) -> str:
    header = "%fused_computation.2186.clone.clone.clone ("
    start = value.index(header)
    end = value.index("\n}\n", start)
    computation = value[start:end]
    low = "padding=0_0x0_64"
    high = "padding=0_0x64_0"
    assert computation.count(low) == 1
    assert computation.count(high) == 1
    swapped = computation.replace(low, "padding=POSITION113_SWAP", 1)
    swapped = swapped.replace(high, low, 1)
    swapped = swapped.replace("padding=POSITION113_SWAP", high, 1)
    return value[:start] + swapped + value[end:]


def _negate_position113_field7_predicate(value: str) -> str:
    anchor = "  %broadcast_select_fusion.218 = "
    start = value.index(anchor)
    end = value.index("\n", start)
    caller = value[start:end]
    original = "fusion(%get-tuple-element.18356, %get-tuple-element.17770, %eq.941)"
    assert caller.count(original) == 1
    attacked = caller.replace(
        original,
        "fusion(%get-tuple-element.18356, %get-tuple-element.17770, "
        "%observer_attack_not)",
        1,
    )
    negation = "  %observer_attack_not = pred[]{:T(512)} not(%eq.941)\n"
    return value[:start] + negation + attacked + value[end:]


def _negate_position113_count_predicate(value: str) -> str:
    anchor = "  %broadcast_in_dim.4828 = "
    start = value.index(anchor)
    end = value.index("\n", start)
    convert = value[start:end]
    assert convert.count("convert(%bitcast.6183)") == 1
    attacked = convert.replace(
        "convert(%bitcast.6183)",
        "convert(%observer_count_not)",
        1,
    )
    negation = (
        "  %observer_count_not = pred[1]{0:T(512)(128)(4,1)} not(%bitcast.6183)\n"
    )
    return value[:start] + negation + attacked + value[end:]


def _replace_position113_current_source_with_named_prior_copy(value: str) -> str:
    original = (
        "%get-tuple-element.18356 = "
        "bf16[1,6144]{1,0:T(2,128)(2,1)} "
        "get-tuple-element(%while.594), index=4"
    )
    attacked = (
        "%get-tuple-element.18356 = "
        "bf16[1,6144]{1,0:T(2,128)(2,1)} "
        "copy(%get-tuple-element.17770)"
    )
    assert value.count(original) == 1
    return value.replace(original, attacked, 1)


@pytest.mark.skipif(
    not REAL_POSITION113_OPTIMIZED_HLO.is_file(),
    reason="protected position-113 PP16 HLO is unavailable",
)
@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            "/*index=15*/%bitcast.6728,",
            "/*index=15*/%copy.1223,",
            1,
        ),
        lambda value: value.replace("constant(113)", "constant(114)"),
        lambda value: value.replace(
            "%all_gather.491, %broadcast_select_fusion.218,",
            "%all_gather.491, %get-tuple-element.17770,",
            1,
        ),
        lambda value: value.replace(
            "/*index=15*/%add.14031,",
            "/*index=15*/%get-tuple-element.17778,",
            1,
        ),
        lambda value: value.replace(
            "fusion(%get-tuple-element.18356, %get-tuple-element.17770, %eq.941)",
            "fusion(%get-tuple-element.17770, %get-tuple-element.17770, %eq.941)",
            1,
        ),
        lambda value: value.replace(
            "add(%get-tuple-element.17778, %broadcast_in_dim.4828)",
            "add(%get-tuple-element.17778, %select_n.5648)",
            1,
        ),
        _swap_position113_current_key_padding_ranges,
        _negate_position113_field7_predicate,
        _negate_position113_count_predicate,
        _replace_position113_current_source_with_named_prior_copy,
    ),
)
def test_feature2_position113_lineage_refuses_mutation_even_with_rebased_identity(
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    original = REAL_POSITION113_OPTIMIZED_HLO.read_text()
    attacked = mutation(original)
    assert attacked != original
    _, identity = canonicalize_feature2_optimized_hlo(attacked)
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_CANONICAL_SHA256",
        identity["sha256"],
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_CANONICAL_BYTES",
        identity["byte_count"],
    )
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_POSITION113_OPTIMIZED_STACK_FRAME_REFERENCES",
        identity["stripped_stack_frame_references"],
    )
    with pytest.raises(BenchmarkValidationError, match="position-113"):
        validate_feature2_main_optimized_hlo(
            attacked,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
        )


def _validate_real_sealed_archive(
    stablehlo_path: Path,
    optimized_hlo_path: Path,
    canonical_hlo_path: Path,
) -> dict[str, object]:
    return validate_feature2_sealed_hlo_archive_identity(
        stablehlo_path,
        optimized_hlo_path,
        canonical_hlo_path,
        expected_stablehlo_sha256=(
            "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4"
        ),
        expected_canonical_sha256=(
            "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
        ),
        expected_canonical_bytes=6_558_627,
        expected_canonicalizer_version=1,
        expected_stripped_stack_frame_references=14_561,
    )


@pytest.mark.skipif(
    not REAL_SEALED_STABLEHLO.is_file() or not REAL_SEALED_OPTIMIZED_HLO.is_file(),
    reason="protected sealed PP16 HLO is unavailable",
)
def test_feature2_real_sealed_archive_recomputes_actual_files(
    tmp_path: Path,
) -> None:
    canonical, _ = canonicalize_feature2_optimized_hlo(
        REAL_SEALED_OPTIMIZED_HLO.read_text()
    )
    canonical_path = tmp_path / "feature2_main.execution_canonical_hlo.txt"
    canonical_path.write_text(canonical)
    report = _validate_real_sealed_archive(
        REAL_SEALED_STABLEHLO,
        REAL_SEALED_OPTIMIZED_HLO,
        canonical_path,
    )
    assert report["stablehlo_sha256"] == (
        "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4"
    )
    assert report["canonical_hlo_identity"]["sha256"] == (
        "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
    )


@pytest.mark.skipif(
    not REAL_SEALED_STABLEHLO.is_file()
    or not REAL_SEALED_OPTIMIZED_HLO.is_file()
    or not REAL_ACQUIRED_MAIN_STABLEHLO.is_file()
    or not REAL_ACQUIRED_MAIN_HLO.is_file(),
    reason="protected sealed/stale PP16 HLO pair is unavailable",
)
@pytest.mark.parametrize(
    "crosswire",
    ("stale_stablehlo", "stale_optimized_hlo", "stale_canonical_hlo"),
)
def test_feature2_real_sealed_archive_refuses_cross_wired_actual_files(
    tmp_path: Path,
    crosswire: str,
) -> None:
    current_canonical, _ = canonicalize_feature2_optimized_hlo(
        REAL_SEALED_OPTIMIZED_HLO.read_text()
    )
    stale_canonical, _ = canonicalize_feature2_optimized_hlo(
        REAL_ACQUIRED_MAIN_HLO.read_text()
    )
    stablehlo_path = REAL_SEALED_STABLEHLO
    optimized_hlo_path = REAL_SEALED_OPTIMIZED_HLO
    canonical = current_canonical
    if crosswire == "stale_stablehlo":
        stablehlo_path = REAL_ACQUIRED_MAIN_STABLEHLO
    elif crosswire == "stale_optimized_hlo":
        optimized_hlo_path = REAL_ACQUIRED_MAIN_HLO
        canonical = stale_canonical
    else:
        canonical = stale_canonical
    canonical_path = tmp_path / "feature2_main.execution_canonical_hlo.txt"
    canonical_path.write_text(canonical)
    with pytest.raises(BenchmarkValidationError):
        _validate_real_sealed_archive(
            stablehlo_path,
            optimized_hlo_path,
            canonical_path,
        )


@pytest.mark.skipif(
    not REAL_SEALED_OPTIMIZED_HLO.is_file(),
    reason="protected sealed PP16 HLO is unavailable",
)
@pytest.mark.parametrize(
    "attack",
    ("dead_decoy", "same_shape_bypass", "root_operand_swap", "mixed_fusion"),
)
def test_feature2_real_sealed_hlo_refuses_any_producer_identity_mutation(
    attack: str,
) -> None:
    original = REAL_SEALED_OPTIMIZED_HLO.read_text()
    root_marker = "\n  ROOT %tuple.2849"
    assert original.count(root_marker) == 1
    if attack == "dead_decoy":
        attacked = original.replace(
            root_marker,
            "\n  %sealed.dead.decoy = bf16[1,1,6144] copy(%copy.1106)" + root_marker,
            1,
        )
    elif attack == "same_shape_bypass":
        attacked = original.replace(
            root_marker,
            "\n  %sealed.same.shape = bf16[1,1,6144] copy(%copy.1106)" + root_marker,
            1,
        ).replace(
            "tuple(%copy-done.105, %min.728, %compare_select_fusion.131, "
            "%bitcast.6914, %maximum_bitcast_fusion.8, /*index=5*/"
            "%reshape_transpose.236, %copy.1106, %bitcast.6876",
            "tuple(%copy-done.105, %min.728, %compare_select_fusion.131, "
            "%bitcast.6914, %maximum_bitcast_fusion.8, /*index=5*/"
            "%reshape_transpose.236, %sealed.same.shape, %bitcast.6876",
            1,
        )
    elif attack == "root_operand_swap":
        attacked = original.replace(
            "%bitcast.6895, %broadcast_in_dim_reshape_transpose.7, /*index=10*/",
            "%broadcast_in_dim_reshape_transpose.7, %bitcast.6895, /*index=10*/",
            1,
        )
    else:
        attacked = original.replace(
            root_marker,
            "\n  %sealed.mixed = bf16[1,1,6144] "
            "fusion(%copy.1106, %bitcast.6876), kind=kLoop, "
            "calls=%fused_computation.0" + root_marker,
            1,
        ).replace(
            "tuple(%copy-done.105, %min.728, %compare_select_fusion.131, "
            "%bitcast.6914, %maximum_bitcast_fusion.8, /*index=5*/"
            "%reshape_transpose.236, %copy.1106, %bitcast.6876",
            "tuple(%copy-done.105, %min.728, %compare_select_fusion.131, "
            "%bitcast.6914, %maximum_bitcast_fusion.8, /*index=5*/"
            "%reshape_transpose.236, %sealed.mixed, %bitcast.6876",
            1,
        )
    assert attacked != original
    with pytest.raises(
        BenchmarkValidationError,
        match="sealed optimized canonical-HLO identity drifted",
    ):
        validate_feature2_main_optimized_hlo(
            attacked,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


@pytest.mark.skipif(
    not REAL_NUMERICAL_REFUSAL_MAIN_HLO.is_file(),
    reason="protected PP16 HLO is unavailable",
)
def test_feature2_real_hlo_refuses_frontend_stack_frame_exchange() -> None:
    original = REAL_NUMERICAL_REFUSAL_MAIN_HLO.read_text()
    attacked = original.replace(" stack_frame_id=350}", "}", 1).replace(
        "frontend_attributes={kernel_metadata={}}",
        "frontend_attributes={kernel_metadata={} stack_frame_id=999999}",
        1,
    )
    assert attacked != original
    assert validate_feature2_main_optimized_hlo(attacked)["passed"] is True
    with pytest.raises(BenchmarkValidationError):
        canonicalize_feature2_optimized_hlo(attacked)


def _materializer_hlo(phase: str) -> str:
    if phase == "query_fp32":
        parameters = """  %b0 = u8[1,2048,2048] parameter(0)
  %s0 = f32[1,16,16] parameter(1)
  %b1 = u8[1,2048,2048] parameter(2)
  %s1 = f32[1,16,16] parameter(3)
  %o0 = f32[1,2048,2048] convert(%b0)
  %o1 = f32[1,2048,2048] convert(%b1)
"""
        root = "(f32[1,2048,2048], f32[1,2048,2048]) tuple(%o0, %o1)"
    elif phase == "wk_decode_bf16":
        parameters = """  %b0 = u8[1,128,6144] parameter(0)
  %s0 = f32[1,1,48] parameter(1)
  %b1 = u8[1,128,6144] parameter(2)
  %s1 = f32[1,1,48] parameter(3)
  %o0 = bf16[1,128,6144] convert(%b0)
  %o1 = bf16[1,128,6144] convert(%b1)
"""
        root = "(bf16[1,128,6144], bf16[1,128,6144]) tuple(%o0, %o1)"
    else:
        parameters = """  %b0 = bf16[1,128,6144] parameter(0)
  %b1 = bf16[1,128,6144] parameter(1)
  %o0 = f32[1,128,6144] convert(%b0)
  %o1 = f32[1,128,6144] convert(%b1)
"""
        root = "(f32[1,128,6144], f32[1,128,6144]) tuple(%o0, %o1)"
    return (
        "HloModule feature2_materializer, num_partitions=2\n\n"
        "ENTRY %main {\n"
        f"{parameters}"
        f"  ROOT %root = {root}\n"
        "}\n"
    )


@pytest.mark.parametrize("phase", ("query_fp32", "wk_decode_bf16", "wk_promote_fp32"))
def test_feature2_materializer_hlo_is_owner_local(phase: str) -> None:
    report = validate_feature2_materializer_optimized_hlo(
        _materializer_hlo(phase), phase=phase
    )
    assert report["passed"] is True
    assert report["num_partitions"] == 2
    assert report["collective_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions=2", "num_partitions=32"),
        lambda value: value.replace(
            "  ROOT %root",
            "  %bad = f32[1] all-reduce(%o0), replica_groups={{0,1}}\n  ROOT %root",
        ),
        lambda value: value.replace(
            "  ROOT %root",
            '  %bad = f32[1] custom-call(), custom_call_target="tpu_custom_call"\n'
            "  ROOT %root",
        ),
        lambda value: value + "host_callback",
    ),
)
def test_feature2_materializer_hlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_materializer_optimized_hlo(
            mutation(_materializer_hlo("query_fp32")), phase="query_fp32"
        )


def _main_optimized_hlo(
    *,
    full_width_rounded_then_slice: bool = False,
    sealed_boundary_capture: bool = False,
    observe_position_113: bool = False,
) -> str:
    assert not observe_position_113 or sealed_boundary_capture
    width = 6144 if full_width_rounded_then_slice else 3072
    attention_per_chunk = 16 if full_width_rounded_then_slice else 32
    down_per_chunk = 16 if full_width_rounded_then_slice else 32
    kernel = f"greenfield_fp8_strategy_nd_o_m8_k512_n{width}"
    computations = [
        "%sum_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {",
        "  %lhs = bf16[] parameter(0)",
        "  %rhs = bf16[] parameter(1)",
        "  ROOT %sum = bf16[] add(%lhs, %rhs)",
        "}",
        "",
    ]

    def half_reducer(
        *,
        prefix: str,
        half0_leaves: list[str],
        half1_leaves: list[str],
        scope: str,
    ) -> tuple[list[str], str]:
        assert len(half0_leaves) == len(half1_leaves) == 16
        body = [f"  %{prefix}.zero = bf16[] constant(0)"]
        reduced_halves = []
        for half, leaves in enumerate((half0_leaves, half1_leaves)):
            stack = f"%{prefix}.stack{half}"
            reshape = f"%{prefix}.reshape{half}"
            y = f"%{prefix}.y{half}"
            body.extend(
                [
                    f"  {stack} = bf16[16,1,3072] concatenate({', '.join(leaves)}), dimensions={{0}}",
                    f"  {reshape} = bf16[4,4,1,3072] reshape({stack})",
                    f"  {y} = bf16[4,1,3072] reduce({reshape}, %{prefix}.zero), dimensions={{1}}, to_apply=%sum_bf16",
                    f"  %{prefix}.y{half}.1024 = bf16[4,1,1024] slice({y}), slice={{[0:4],[0:1],[0:1024]}}",
                    f"  %{prefix}.y{half}.2048 = bf16[4,1,2048] slice({y}), slice={{[0:4],[0:1],[1024:3072]}}",
                    f"  %{prefix}.add{half}.1024 = bf16[4,1,1024] add(%{prefix}.y{half}.1024, %{prefix}.y{half}.1024)",
                    f"  %{prefix}.add{half}.2048 = bf16[4,1,2048] add(%{prefix}.y{half}.2048, %{prefix}.y{half}.2048)",
                    f"  %{prefix}.combined{half} = bf16[4,1,3072] concatenate(%{prefix}.add{half}.1024, %{prefix}.add{half}.2048), dimensions={{2}}",
                ]
            )
            reduced_halves.append(f"%{prefix}.combined{half}")
        body.extend(
            [
                f"  %{prefix}.partition = u32[] partition-id()",
                f"  %{prefix}.partition_mask = u32[] constant(1)",
                f"  %{prefix}.local_partition = u32[] and(%{prefix}.partition, %{prefix}.partition_mask)",
                f"  %{prefix}.local_partition_s32 = s32[] convert(%{prefix}.local_partition)",
                f"  %{prefix}.zero_partition = s32[] constant(0)",
                f"  %{prefix}.owns_half0 = pred[] compare(%{prefix}.local_partition_s32, %{prefix}.zero_partition), direction=EQ",
                f'  %{prefix}.select = bf16[4,1,3072] select(%{prefix}.owns_half0, {reduced_halves[1]}, {reduced_halves[0]}), metadata={{op_name="jit(f)/{scope}/jit(_where)/select_n"}}',
                f'  %{prefix}.permute = bf16[4,1,3072] collective-permute(%{prefix}.select), source_target_pairs={{{{0,1}},{{1,0}}}}, metadata={{op_name="jit(f)/{scope}/greenfield_strategy_nd_feature2_lp2_x_exchange/ppermute"}}',
            ]
        )
        return body, f"%{prefix}.permute"

    for chunk in range(4):
        body = [
            f"%chunk{chunk} () -> (bf16[4,1,3072], bf16[4,1,3072]) {{",
            "  %ain = bf16[8,512] constant(0)",
            f"  %aw = u8[{width},512] constant(0)",
            "  %as = f32[8,128] constant(0)",
            "  %gate_lhs = bf16[1,6144] constant(0)",
            "  %gate_rhs = bf16[6144,768] constant(0)",
            f"  %down_rhs = bf16[384,{width}] constant(0)",
        ]
        attention_half0: list[str] = []
        attention_half1: list[str] = []
        for index in range(attention_per_chunk):
            name = f"%att.{chunk}.{index}"
            body.append(
                f"  {name} = bf16[8,{width}] custom-call(%ain, %aw, %as), "
                'custom_call_target="tpu_custom_call", '
                f'kernel_name="{kernel}", metadata={{op_name="jit(f)/{kernel}/pallas_call"}}'
            )
            row = f"%att.row.{chunk}.{index}"
            body.append(
                f"  {row} = bf16[1,{width}] slice({name}), slice={{[0:1],[0:{width}]}}"
            )
            if full_width_rounded_then_slice:
                lower = f"%att.lower.{chunk}.{index}"
                upper = f"%att.upper.{chunk}.{index}"
                body.extend(
                    [
                        f"  {lower} = bf16[1,3072] slice({row}), slice={{[0:1],[0:3072]}}",
                        f"  {upper} = bf16[1,3072] slice({row}), slice={{[0:1],[3072:6144]}}",
                    ]
                )
                candidates = ((attention_half0, lower), (attention_half1, upper))
            else:
                candidates = (
                    (attention_half0 if index < 16 else attention_half1, row),
                )
            for destination, value in candidates:
                half = 0 if destination is attention_half0 else 1
                leaf = f"%att.leaf.{chunk}.{half}.{len(destination)}"
                body.append(f"  {leaf} = bf16[1,1,3072] reshape({value})")
                destination.append(leaf)
        for index in range(16):
            name = f"%gate.{chunk}.{index}"
            body.append(
                f"  {name} = f32[1,768] convolution(%gate_lhs, %gate_rhs), "
                'dim_labels=bf_io->bf, metadata={op_name="jit(f)/'
                f"greenfield_dense_feature2_virtual_rank_{index:02d}/"
                'conv_general_dilated"}'
            )
        dense_half0: list[str] = []
        dense_half1: list[str] = []
        for index in range(down_per_chunk):
            name = f"%down.{chunk}.{index}"
            scope = (
                "greenfield_dense_feature2_full_width_then_slice"
                if full_width_rounded_then_slice
                else f"greenfield_dense_feature2_half_{index % 2}"
            )
            gate_slice = f"%gate.slice.{chunk}.{index}"
            activated = f"%activated.{chunk}.{index}"
            rounded = f"%down.rounded.{chunk}.{index}"
            body.extend(
                [
                    f"  {gate_slice} = f32[1,384] slice(%gate.{chunk}.{index % 16}), slice={{[0:1],[0:384]}}",
                    f"  {activated} = bf16[1,384] convert({gate_slice})",
                    f'  {name} = f32[1,{width}] convolution({activated}, %down_rhs), dim_labels=bf_io->bf, metadata={{op_name="jit(f)/greenfield_dense_feature2_virtual_rank_{index % 16:02d}/{scope}/conv_general_dilated"}}',
                    f"  {rounded} = bf16[1,{width}] convert({name})",
                ]
            )
            if full_width_rounded_then_slice:
                lower = f"%down.lower.{chunk}.{index}"
                upper = f"%down.upper.{chunk}.{index}"
                body.extend(
                    [
                        f"  {lower} = bf16[1,3072] slice({rounded}), slice={{[0:1],[0:3072]}}",
                        f"  {upper} = bf16[1,3072] slice({rounded}), slice={{[0:1],[3072:6144]}}",
                    ]
                )
                candidates = ((dense_half0, lower), (dense_half1, upper))
            else:
                candidates = ((dense_half0 if index < 16 else dense_half1, rounded),)
            for destination, value in candidates:
                half = 0 if destination is dense_half0 else 1
                leaf = f"%down.leaf.{chunk}.{half}.{len(destination)}"
                body.append(f"  {leaf} = bf16[1,1,3072] reshape({value})")
                destination.append(leaf)
        attention_reducer, attention_output = half_reducer(
            prefix=f"attention.{chunk}",
            half0_leaves=attention_half0,
            half1_leaves=attention_half1,
            scope="greenfield_strategy_nd_feature2_attention_output",
        )
        dense_reducer, dense_output = half_reducer(
            prefix=f"dense.{chunk}",
            half0_leaves=dense_half0,
            half1_leaves=dense_half1,
            scope="greenfield_strategy_nd_feature2_dense_down",
        )
        body.extend(attention_reducer)
        body.extend(dense_reducer)
        body.extend(
            [
                (
                    "  ROOT %chunk_root = "
                    "(bf16[4,1,3072], bf16[4,1,3072]) "
                    f"tuple({attention_output}, {dense_output})"
                ),
                "}",
                "",
            ]
        )
        computations.extend(body)

    root_fields = [
        "s32[1,2048]",
        "s32[1]",
        "f32[1,2048]",
        "bf16[1,1,3072]",
        "bf16[1,1,32,256]",
        "bf16[1,576]",
        "bf16[1,16,256,640]",
        "bf16[1,16,256,128]",
        "bf16[1,16,256,128]",
        "u32[1,2]",
        "pred[1]",
    ]
    if sealed_boundary_capture:
        root_fields[6:6] = [
            "bf16[1,1,6144]",
            "bf16[1,1,2048]",
            "f32[1,1,32,128]",
            "f32[1,1,32]",
        ]
    root = f"({', '.join(root_fields)})"
    lines = [
        "HloModule feature2_main, num_partitions=2",
        "",
        *computations,
        "ENTRY %main {",
        "  %tokens = s32[8156] parameter(0)",
        "  %positions = s32[8156] parameter(1)",
        "  %blocks = s32[1,16] parameter(2)",
        "  %context = s32[1] parameter(3)",
        "  %position = s32[1] parameter(4)",
        "  %rope = bf16[8192,64] parameter(5)",
        "  %g = bf16[1,3072] all-gather(%rope), replica_groups={{0,1}}, use_global_device_ids=true",
        "  %p = bf16[1,3072] collective-permute(%g), source_target_pairs={{0,1},{1,0}}",
    ]
    for index in range(8):
        lines.append(
            f"  %attn.{index} = bf16[1,16,256] custom-call(%rope), "
            'custom_call_target="tpu_custom_call", '
            'metadata={op_name="greenfield_pregathered_sparse_mla_'
            'h16_k2048_b512_w640/pallas_call"}'
        )
    for chunk in range(4):
        lines.extend(
            [
                f"  %chunk_call.{chunk} = (bf16[4,1,3072], bf16[4,1,3072]) fusion(), calls=%chunk{chunk}",
                f"  %chunk.attention.{chunk} = bf16[4,1,3072] get-tuple-element(%chunk_call.{chunk}), index=0",
                f"  %chunk.dense.{chunk} = bf16[4,1,3072] get-tuple-element(%chunk_call.{chunk}), index=1",
                f"  %chunk.sum.{chunk} = bf16[4,1,3072] add(%chunk.attention.{chunk}, %chunk.dense.{chunk})",
                f"  %chunk.row.{chunk} = bf16[1,1,3072] slice(%chunk.sum.{chunk}), slice={{[0:1],[0:1],[0:3072]}}",
            ]
        )
    lines.extend(
        [
            "  %chunks.01 = bf16[1,1,3072] add(%chunk.row.0, %chunk.row.1)",
            "  %chunks.23 = bf16[1,1,3072] add(%chunk.row.2, %chunk.row.3)",
            "  %residual = bf16[1,1,3072] add(%chunks.01, %chunks.23)",
            "  %out.positions = s32[1,2048] fusion(%tokens)",
            "  %out.valid = s32[1] copy(%context)",
            "  %out.scores = f32[1,2048] fusion(%positions)",
            "  %out.attention = bf16[1,1,32,256] fusion("
            + ", ".join(f"%attn.{index}" for index in range(8))
            + ")",
            "  %out.kv = bf16[1,576] fusion(%rope)",
            "  %out.kv_cache = bf16[1,16,256,640] fusion(%p)",
            "  %out.index0 = bf16[1,16,256,128] fusion(%blocks)",
            "  %out.index1 = bf16[1,16,256,128] fusion(%position)",
            "  %out.digest = u32[1,2] fusion(%position)",
            "  %out.contract = pred[1] compare(%context, %position), direction=EQ",
        ]
    )
    root_operands = [
        "%out.positions",
        "%out.valid",
        "%out.scores",
        "%residual",
        "%out.attention",
        "%out.kv",
        "%out.kv_cache",
        "%out.index0",
        "%out.index1",
        "%out.digest",
        "%out.contract",
    ]
    if sealed_boundary_capture:
        sealed = (
            ("normalized", "bf16[1,1,6144]", "normalized_hidden"),
            ("q_a", "bf16[1,1,2048]", "q_a_state"),
            ("query", "f32[1,1,32,128]", "dsa_query"),
            ("head", "f32[1,1,32]", "dsa_head_weights"),
        )
        for short_name, shape, marker in sealed:
            lines.extend(
                [
                    f"  %sealed.{short_name}.source = {shape} fusion(%residual)",
                    (
                        f"  %sealed.{short_name} = {shape} optimization-barrier("
                        f'%sealed.{short_name}.source), metadata={{op_name="jit(main)/'
                        f"greenfield_pp16_feature2_sealed_{marker}/"
                        'optimization_barrier"}'
                    ),
                ]
            )
        root_operands[6:6] = [
            "%sealed.normalized",
            "%sealed.q_a",
            "%sealed.query",
            "%sealed.head",
        ]
    if observe_position_113:
        observed = (
            ("normalized", "bf16[1,1,6144]", "normalized_hidden"),
            ("q_a", "bf16[1,1,2048]", "q_a_state"),
            ("query", "f32[1,1,32,128]", "dsa_query"),
            ("head", "f32[1,1,32]", "dsa_head_weights"),
            ("key", "f32[1,1,128]", "current_key"),
            ("positions", "s32[1,1,2048]", "selected_positions"),
            ("valid", "s32[1,1]", "selected_valid_counts"),
            ("scores", "f32[1,1,2048]", "selected_scores"),
            ("count", "s32[1,1]", "observation_count"),
        )
        for short_name, shape, marker in observed:
            lines.extend(
                [
                    f"  %p113.{short_name}.source = {shape} fusion(%residual)",
                    (
                        f"  %p113.{short_name} = {shape} optimization-barrier("
                        f'%p113.{short_name}.source), metadata={{op_name="jit(main)/'
                        f"greenfield_pp16_feature2_p113_{marker}/"
                        'optimization_barrier"}'
                    ),
                ]
            )
        root_fields.extend(shape for _, shape, _ in observed)
        root_operands.extend(f"%p113.{name}" for name, _, _ in observed)
        root = f"({', '.join(root_fields)})"
    lines.extend(
        [
            f"  ROOT %root = {root} tuple({', '.join(root_operands)})",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def test_feature2_main_optimized_hlo_is_parsed_and_lp2_only() -> None:
    report = validate_feature2_main_optimized_hlo(_main_optimized_hlo())
    assert report["passed"] is True
    assert report["h16_b512_attention_calls"] == 8
    assert report["live_attention_producer_count"] == 128
    assert report["live_dense_down_convolution_count"] == 128
    assert report["physical_collective_count"] == 10
    assert report["collective_counts"] == {
        "all-gather": 1,
        "collective-permute": 9,
    }


def test_feature2_main_successor_optimized_hlo_pins_live_full_width_producers() -> None:
    report = validate_feature2_main_optimized_hlo(
        _main_optimized_hlo(full_width_rounded_then_slice=True),
        full_width_rounded_then_slice=True,
    )
    assert report["projection_width"] == 6144
    assert report["live_attention_producer_count"] == 64
    assert report["live_dense_down_convolution_count"] == 64
    assert report["reducer_counts"] == {
        "greenfield_strategy_nd_feature2_attention_output": 4,
        "greenfield_strategy_nd_feature2_dense_down": 4,
    }


def test_feature2_main_successor_optimized_hlo_pins_sealed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hlo = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    _pin_synthetic_sealed_optimized_hlo(monkeypatch, hlo)
    report = validate_feature2_main_optimized_hlo(
        hlo,
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    assert report["sealed_boundary_capture"] is True
    assert report["output_count"] == 15
    assert report["sealed_bindings"] == {
        "6": "greenfield_pp16_feature2_sealed_normalized_hidden",
        "7": "greenfield_pp16_feature2_sealed_q_a_state",
        "8": "greenfield_pp16_feature2_sealed_dsa_query",
        "9": "greenfield_pp16_feature2_sealed_dsa_head_weights",
    }
    with pytest.raises(BenchmarkValidationError, match="terminal boundary drifted"):
        validate_feature2_main_optimized_hlo(
            _main_optimized_hlo(full_width_rounded_then_slice=True),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


def test_feature2_main_position113_observer_refuses_geometry_only_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hlo = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    _pin_synthetic_position113_optimized_hlo(monkeypatch, hlo)
    with pytest.raises(BenchmarkValidationError, match="final-while field"):
        validate_feature2_main_optimized_hlo(
            hlo,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
        )


def test_feature2_db518_optimized_hlo_refuses_unpinned_synthetic_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hlo = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    _pin_synthetic_position113_optimized_hlo(monkeypatch, hlo)
    with pytest.raises(BenchmarkValidationError, match="canonical-HLO identity"):
        validate_feature2_main_optimized_hlo(
            hlo,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            exact_layer0_prompt_keys=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions=2", "num_partitions=32", 1),
        lambda value: value.replace(
            "source_target_pairs={{0,1},{1,0}}",
            "source_target_pairs={{0,2},{2,0}}",
            1,
        ),
        lambda value: value + "\nhost_callback",
        lambda value: value + "\nbf16[32,6144]",
    ),
)
def test_feature2_db518_acquisition_keeps_structural_refusals(
    mutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hlo = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    _pin_synthetic_position113_optimized_hlo(monkeypatch, hlo)
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            mutation(hlo),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            exact_layer0_prompt_keys=True,
        )


def test_feature2_position113_optimized_hlo_refuses_wrong_source_under_exact_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    _pin_synthetic_position113_optimized_hlo(monkeypatch, original)
    attacked = original.replace(
        "%p113.normalized.source), metadata=",
        "%sealed.normalized.source), metadata=",
        1,
    )
    assert attacked != original
    with pytest.raises(
        BenchmarkValidationError, match="canonical-HLO identity drifted"
    ):
        validate_feature2_main_optimized_hlo(
            attacked,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            ", %sealed.head, %out.kv_cache", ", %out.kv_cache", 1
        ),
        lambda value: value.replace(
            "%sealed.normalized, %sealed.q_a",
            "%sealed.normalized.source, %sealed.q_a",
            1,
        ),
        lambda value: value.replace(
            "greenfield_pp16_feature2_sealed_normalized_hidden",
            "greenfield_pp16_feature2_sealed_q_a_state",
            1,
        ),
        lambda value: value.replace(
            "%sealed.query, %sealed.head", "%sealed.head, %sealed.query", 1
        ),
        lambda value: value.replace(
            "  ROOT %root",
            "  %sealed.normalized.mixed = bf16[1,1,6144] "
            "fusion(%sealed.normalized, %sealed.normalized.source)\n"
            "  ROOT %root",
            1,
        ).replace(
            "%sealed.normalized, %sealed.q_a",
            "%sealed.normalized.mixed, %sealed.q_a",
            1,
        ),
    ),
)
def test_feature2_main_successor_optimized_hlo_refuses_unbound_sealed_roots(
    mutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _main_optimized_hlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    _pin_synthetic_sealed_optimized_hlo(monkeypatch, original)
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            mutation(original),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


def test_feature2_main_successor_optimized_hlo_refuses_hybrid_and_dead_decoy() -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    hybrid = original.replace(
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
        "greenfield_fp8_strategy_nd_o_m8_k512_n3072",
        2,
    )
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(hybrid, full_width_rounded_then_slice=True)

    missing_live = original.replace(
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
        "greenfield_dead_replaced_attention_o",
        2,
    )
    dead_decoy = """
%dead_decoy (unused: bf16[1]) -> bf16[8,6144] {
  %ain = bf16[8,512] parameter(0)
  %aw = u8[6144,512] parameter(1)
  %as = f32[8,128] parameter(2)
  ROOT %dead = bf16[8,6144] custom-call(%ain, %aw, %as), custom_call_target="tpu_custom_call", kernel_name="greenfield_fp8_strategy_nd_o_m8_k512_n6144", metadata={op_name="jit(dead)/greenfield_fp8_strategy_nd_o_m8_k512_n6144/pallas_call"}
}
"""
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            missing_live + dead_decoy,
            full_width_rounded_then_slice=True,
        )


def test_feature2_main_successor_optimized_hlo_refuses_reducer_or_stack_drift() -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    mutations = (
        original.replace(
            "greenfield_strategy_nd_feature2_attention_output/"
            "greenfield_strategy_nd_feature2_lp2_x_exchange/ppermute",
            "greenfield_unbound_attention_output/"
            "greenfield_strategy_nd_feature2_lp2_x_exchange/ppermute",
            1,
        ),
        original + "\nbf16[16,1,6144] full_leaf_stack\n",
    )
    for mutation in mutations:
        with pytest.raises(BenchmarkValidationError):
            validate_feature2_main_optimized_hlo(
                mutation, full_width_rounded_then_slice=True
            )


def _swap_first_attention_dense_reducer_operands(optimized_hlo: str) -> str:
    lines = optimized_hlo.splitlines(keepends=True)
    selected: dict[str, tuple[int, str]] = {}
    for index, line in enumerate(lines):
        for kind, scope in (
            ("attention", "greenfield_strategy_nd_feature2_attention_output"),
            ("dense", "greenfield_strategy_nd_feature2_dense_down"),
        ):
            if kind in selected or scope not in line or "/ppermute" not in line:
                continue
            match = re.search(
                r"\bcollective-permute(?:-start)?\((%[A-Za-z0-9_.-]+)\)",
                line,
            )
            if match is not None:
                selected[kind] = (index, match.group(1))
    assert set(selected) == {"attention", "dense"}
    attention_index, attention_operand = selected["attention"]
    dense_index, dense_operand = selected["dense"]
    assert attention_operand != dense_operand
    lines[attention_index] = lines[attention_index].replace(
        f"({attention_operand})", f"({dense_operand})", 1
    )
    lines[dense_index] = lines[dense_index].replace(
        f"({dense_operand})", f"({attention_operand})", 1
    )
    return "".join(lines)


def test_feature2_optimized_hlo_refuses_type_valid_scoped_reducer_swap() -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    swapped = _swap_first_attention_dense_reducer_operands(original)
    assert swapped != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            swapped, full_width_rounded_then_slice=True
        )


def _swap_first_successor_stack_leaves(
    optimized_hlo: str,
    *,
    kind: str,
) -> str:
    leaf_prefix, reducer_prefix = {
        "attention": ("att", "attention"),
        "dense": ("down", "dense"),
    }[kind]
    lower = f"%{leaf_prefix}.leaf.0.0.0"
    upper = f"%{leaf_prefix}.leaf.0.1.0"
    lines = optimized_hlo.splitlines(keepends=True)
    changed = 0
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(
            (f"%{reducer_prefix}.0.stack0 =", f"%{reducer_prefix}.0.stack1 =")
        ):
            continue
        if lower in line:
            lines[index] = line.replace(lower, upper, 1)
            changed += 1
        elif upper in line:
            lines[index] = line.replace(upper, lower, 1)
            changed += 1
    assert changed == 2
    return "".join(lines)


def _swap_first_successor_owner_select_branches(
    optimized_hlo: str,
    *,
    kind: str,
) -> str:
    prefix = "attention" if kind == "attention" else "dense"
    lines = optimized_hlo.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(f"%{prefix}.0.select ="):
            continue
        match = re.search(
            r"select\((%[^,]+), (%[^,]+), (%[^)]+)\)",
            line,
        )
        assert match is not None
        predicate, true_value, false_value = match.groups()
        lines[index] = (
            line[: match.start()]
            + f"select({predicate}, {false_value}, {true_value})"
            + line[match.end() :]
        )
        return "".join(lines)
    raise AssertionError(f"missing {kind} owner select")


def _shift_first_successor_owner_zero(
    optimized_hlo: str,
    *,
    kind: str,
) -> str:
    prefix = "attention" if kind == "attention" else "dense"
    zero = f"  %{prefix}.0.zero_partition = s32[] constant(0)\n"
    injected = (
        zero
        + f"  %{prefix}.0.one_partition = s32[] constant(1)\n"
        + f"  %{prefix}.0.shifted_zero = s32[] add("
        f"%{prefix}.0.one_partition, %{prefix}.0.zero_partition)\n"
    )
    compare = (
        f"compare(%{prefix}.0.local_partition_s32, "
        f"%{prefix}.0.zero_partition), direction=EQ"
    )
    shifted_compare = (
        f"compare(%{prefix}.0.local_partition_s32, "
        f"%{prefix}.0.shifted_zero), direction=EQ"
    )
    assert optimized_hlo.count(zero) == 1
    assert optimized_hlo.count(compare) == 1
    return optimized_hlo.replace(zero, injected, 1).replace(compare, shifted_compare, 1)


def _add_first_successor_predicate_metadata_decoy(
    optimized_hlo: str,
    *,
    kind: str,
    attack: str,
) -> str:
    prefix = "attention" if kind == "attention" else "dense"
    if attack == "direction":
        original = f"%{prefix}.0.zero_partition), direction=EQ"
        attacked = (
            f"%{prefix}.0.zero_partition), direction=NE, "
            'metadata={op_name="direction=EQ"}'
        )
    elif attack == "mask":
        original = f"%{prefix}.0.partition_mask = u32[] constant(1)"
        attacked = (
            f"%{prefix}.0.partition_mask = u32[] constant(2), "
            'metadata={op_name="constant(1)"}'
        )
    else:
        raise AssertionError(f"unknown predicate metadata attack: {attack}")
    assert optimized_hlo.count(original) == 1
    return optimized_hlo.replace(original, attacked, 1)


@pytest.mark.parametrize("kind", ("attention", "dense"))
def test_feature2_successor_optimized_hlo_refuses_half_stack_leaf_swap(
    kind: str,
) -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    swapped = _swap_first_successor_stack_leaves(original, kind=kind)
    assert swapped != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            swapped, full_width_rounded_then_slice=True
        )


@pytest.mark.parametrize("kind", ("attention", "dense"))
def test_feature2_successor_optimized_hlo_refuses_owner_select_branch_swap(
    kind: str,
) -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    swapped = _swap_first_successor_owner_select_branches(original, kind=kind)
    assert swapped != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            swapped, full_width_rounded_then_slice=True
        )


@pytest.mark.parametrize("kind", ("attention", "dense"))
def test_feature2_successor_optimized_hlo_refuses_shifted_owner_zero(
    kind: str,
) -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    shifted = _shift_first_successor_owner_zero(original, kind=kind)
    assert shifted != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            shifted, full_width_rounded_then_slice=True
        )


@pytest.mark.parametrize("kind", ("attention", "dense"))
@pytest.mark.parametrize("attack", ("direction", "mask"))
def test_feature2_successor_optimized_hlo_refuses_predicate_metadata_decoy(
    kind: str,
    attack: str,
) -> None:
    original = _main_optimized_hlo(full_width_rounded_then_slice=True)
    attacked = _add_first_successor_predicate_metadata_decoy(
        original,
        kind=kind,
        attack=attack,
    )
    assert attacked != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(
            attacked, full_width_rounded_then_slice=True
        )


@pytest.mark.skipif(
    not REAL_ACQUIRED_MAIN_HLO.is_file(),
    reason="protected PP16 HLO is unavailable",
)
def test_feature2_real_optimized_hlo_refuses_type_valid_reducer_swap() -> None:
    original = REAL_ACQUIRED_MAIN_HLO.read_text()
    swapped = _swap_first_attention_dense_reducer_operands(original)
    assert swapped != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(swapped)


def test_feature_shard_axis_survives_in_real_optimized_root() -> None:
    code = r"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

mesh = Mesh(np.asarray(jax.devices()), ("feature",))
mapped = jax.shard_map(
    lambda value: value,
    mesh=mesh,
    in_specs=P("feature", None, None),
    out_specs=P("feature", None, None),
    check_vma=False,
)
optimized = jax.jit(mapped).lower(
    jnp.zeros((2, 1, 3072), dtype=jnp.bfloat16)
).compile().as_text()
module = parse_hlo_module(optimized)
roots = [
    item for item in module.instructions
    if item.computation.startswith("ENTRY ")
    and item.raw_line.startswith("ROOT ")
]
assert len(roots) == 1
assert tuple(
    (shape.dtype.lower(), shape.dimensions)
    for shape in roots[0].result_shapes
) == (("bf16", (1, 1, 3072)),)
"""
    environment = os.environ.copy()
    existing = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=2".strip()
    )
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(Path.cwd())
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions=2", "num_partitions=32"),
        lambda value: value.replace("{{0,1}}", "{{0,1,2,3}}"),
        lambda value: value.replace(", use_global_device_ids=true", ""),
        lambda value: value.replace("{{0,1},{1,0}}", "{{0,2},{2,0}}"),
        lambda value: value.replace(
            "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640",
            "greenfield_pregathered_sparse_mla_h32_k2048_b512_w640",
            1,
        ),
        lambda value: value + "\nbf16[8156,6144] host_callback",
    ),
)
def test_feature2_main_optimized_hlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(mutation(_main_optimized_hlo()))


def _main_stablehlo(
    *,
    full_width_rounded_then_slice: bool = False,
    sealed_boundary_capture: bool = False,
    observe_position_113: bool = False,
) -> str:
    assert not observe_position_113 or sealed_boundary_capture
    width = 6144 if full_width_rounded_then_slice else 3072
    attention_per_chunk = 16 if full_width_rounded_then_slice else 32
    down_per_chunk = 16 if full_width_rounded_then_slice else 32
    kernel = f"greenfield_fp8_strategy_nd_o_m8_k512_n{width}"
    chunks = []
    for chunk in range(4):
        body = [
            (
                f"  func.func private @chunk{chunk}(%arg0: tensor<8x512xbf16>, "
                f"%arg1: tensor<{width}x512xui8>, %arg2: tensor<8x128xf32>, "
                "%gate_lhs: tensor<1x6144xbf16>, "
                "%gate_rhs: tensor<6144x768xbf16>, "
                "%down_lhs: tensor<1x384xbf16>, "
                f"%down_rhs: tensor<384x{width}xbf16>) -> "
                "tensor<1x3072xbf16> {"
            ),
        ]
        first_half = None
        for index in range(attention_per_chunk):
            producer = f"%a{chunk}_{index}"
            row = f"%ar{chunk}_{index}"
            body.append(
                f"    {producer} = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1, %arg2) "
                f'{{kernel_name = "{kernel}"}} : (tensor<8x512xbf16>, '
                f"tensor<{width}x512xui8>, tensor<8x128xf32>) -> tensor<8x{width}xbf16>"
            )
            body.append(
                f"    {row} = stablehlo.slice {producer} [0:1, 0:{width}] : "
                f"(tensor<8x{width}xbf16>) -> tensor<1x{width}xbf16>"
            )
            if full_width_rounded_then_slice:
                half0 = f"%ah0_{chunk}_{index}"
                half1 = f"%ah1_{chunk}_{index}"
                body.extend(
                    [
                        (
                            f"    {half0} = stablehlo.slice {row} "
                            "[0:1, 0:3072] : (tensor<1x6144xbf16>) -> "
                            "tensor<1x3072xbf16>"
                        ),
                        (
                            f"    {half1} = stablehlo.slice {row} "
                            "[0:1, 3072:6144] : (tensor<1x6144xbf16>) -> "
                            "tensor<1x3072xbf16>"
                        ),
                    ]
                )
                first_half = first_half or half0
            else:
                first_half = first_half or row
        for index in range(16):
            gate = f"%g{chunk}_{index}"
            body.extend(
                [
                    (
                        f"    {gate} = stablehlo.convolution"
                        "(%gate_lhs, %gate_rhs) : (tensor<1x6144xbf16>, "
                        "tensor<6144x768xbf16>) -> tensor<1x768xf32>"
                    ),
                    (
                        f"    %gr{chunk}_{index} = stablehlo.convert {gate} : "
                        "(tensor<1x768xf32>) -> tensor<1x768xbf16>"
                    ),
                ]
            )
        for index in range(down_per_chunk):
            down = f"%d{chunk}_{index}"
            rounded = f"%dr{chunk}_{index}"
            body.extend(
                [
                    (
                        f"    {down} = stablehlo.convolution"
                        "(%down_lhs, %down_rhs) : (tensor<1x384xbf16>, "
                        f"tensor<384x{width}xbf16>) -> tensor<1x{width}xf32>"
                    ),
                    (
                        f"    {rounded} = stablehlo.convert {down} : "
                        f"(tensor<1x{width}xf32>) -> tensor<1x{width}xbf16>"
                    ),
                ]
            )
            if full_width_rounded_then_slice:
                body.extend(
                    [
                        (
                            f"    %dh0_{chunk}_{index} = stablehlo.slice "
                            f"{rounded} [0:1, 0:3072] : "
                            "(tensor<1x6144xbf16>) -> tensor<1x3072xbf16>"
                        ),
                        (
                            f"    %dh1_{chunk}_{index} = stablehlo.slice "
                            f"{rounded} [0:1, 3072:6144] : "
                            "(tensor<1x6144xbf16>) -> tensor<1x3072xbf16>"
                        ),
                    ]
                )
        body.extend(
            [
                (
                    f'    %p{chunk}_0 = "stablehlo.collective_permute"'
                    f"({first_half}) <{{source_target_pairs = "
                    "dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : "
                    "(tensor<4x1x3072xbf16>) -> tensor<4x1x3072xbf16>"
                ),
                (
                    f'    %p{chunk}_1 = "stablehlo.collective_permute"'
                    f"({first_half}) <{{source_target_pairs = "
                    "dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : "
                    "(tensor<4x1x3072xbf16>) -> tensor<4x1x3072xbf16>"
                ),
                f"    return {first_half} : tensor<1x3072xbf16>",
                "  }",
            ]
        )
        chunks.extend(body)
    main_results = [
        ("tensor<1x2048xi32>", "result.event1_positions"),
        ("tensor<1xi32>", "result.event1_valid_counts"),
        ("tensor<1x2048xf32>", "result.event1_scores"),
        ("tensor<2x1x3072xbf16>", "result.current_carried_halves"),
        (
            "tensor<2x1x32x256xbf16>",
            "result.current_attention_query_owners",
        ),
        ("tensor<1x576xbf16>", "result.current_kv_a"),
        (
            "tensor<2x16x256x640xbf16>",
            "result.layer0_kv_cache_owners",
        ),
        (
            "tensor<2x16x256x128xbf16>",
            "result.layer0_index_cache_owners",
        ),
        (
            "tensor<2x16x256x128xbf16>",
            "result.layer1_index_cache_owners",
        ),
        ("tensor<2x2xui32>", "result.carried_liveness_digest_owners"),
        ("tensor<1xi1>", "result.contract_valid"),
    ]
    if sealed_boundary_capture:
        main_results[6:6] = [
            (
                "tensor<2x1x6144xbf16>",
                "result.current_normalized_hidden_owners",
            ),
            ("tensor<2x1x2048xbf16>", "result.current_q_a_state_owners"),
            ("tensor<2x1x32x128xf32>", "result.current_dsa_query_owners"),
            (
                "tensor<2x1x32xf32>",
                "result.current_dsa_head_weights_owners",
            ),
        ]
    if observe_position_113:
        main_results.extend(
            [
                (
                    "tensor<2x1x6144xbf16>",
                    "result.position113_normalized_hidden_owners",
                ),
                (
                    "tensor<2x1x2048xbf16>",
                    "result.position113_q_a_state_owners",
                ),
                (
                    "tensor<2x1x32x128xf32>",
                    "result.position113_dsa_query_owners",
                ),
                (
                    "tensor<2x1x32xf32>",
                    "result.position113_dsa_head_weights_owners",
                ),
                (
                    "tensor<2x1x128xf32>",
                    "result.position113_current_key_owners",
                ),
                (
                    "tensor<2x1x2048xi32>",
                    "result.position113_selected_positions_owners",
                ),
                (
                    "tensor<2x1xi32>",
                    "result.position113_selected_valid_counts_owners",
                ),
                (
                    "tensor<2x1x2048xf32>",
                    "result.position113_selected_scores_owners",
                ),
                (
                    "tensor<2x1xi32>",
                    "result.position113_observation_count_owners",
                ),
            ]
        )
    result_signature = ", ".join(
        f'{tensor_type} {{jax.result_info = "{result_info}"}}'
        for tensor_type, result_info in main_results
    )
    result_types = ", ".join(item[0] for item in main_results)
    result_operands = ", ".join(
        f"%result#{index}" for index in range(len(main_results))
    )
    return (
        f"""module attributes {{mhlo.num_partitions = 2 : i32}} {{
  func.func public @main(%arg0: tensor<8156xi32>, %arg1: tensor<8156xi32>,
      %arg2: tensor<1x16xi32>, %arg3: tensor<8192x64xbf16>)
      -> ({result_signature}) {{
    %0 = stablehlo.all_gather %arg3, dim = 0,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    %1 = stablehlo.collective_permute %0,
      source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>
    %2 = stablehlo.custom_call @tpu_custom_call(%1)
      {{backend_config = "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"}}
    %3 = call @chunk0() : () -> tensor<1x3072xbf16>
    %4 = call @chunk1() : () -> tensor<1x3072xbf16>
    %5 = call @chunk2() : () -> tensor<1x3072xbf16>
    %6 = call @chunk3() : () -> tensor<1x3072xbf16>
    %result:{len(main_results)} = "test.results"() : () -> ({result_types})
    return {result_operands} : {result_types}
  }}
"""
        + "\n".join(chunks)
        + "\n}\n"
    )


def test_feature2_main_stablehlo_pins_boundary_and_groups() -> None:
    assert validate_feature2_main_stablehlo(_main_stablehlo())["passed"] is True


def test_feature2_main_successor_stablehlo_pins_immediate_half_slices() -> None:
    report = validate_feature2_main_stablehlo(
        _main_stablehlo(full_width_rounded_then_slice=True),
        full_width_rounded_then_slice=True,
    )
    assert report["projection_contract"]["attention_producer_count"] == 64
    assert report["projection_contract"]["dense_down_convolution_count"] == 64
    assert report["projection_contract"]["half_reducer_count"] == 8


def test_feature2_main_successor_stablehlo_pins_sealed_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stablehlo = _main_stablehlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    _pin_synthetic_sealed_stablehlo(monkeypatch, stablehlo)
    report = validate_feature2_main_stablehlo(
        stablehlo,
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    assert report["sealed_boundary_capture"] is True
    assert report["output_count"] == 15
    assert report["result_infos"][6:10] == [
        "result.current_normalized_hidden_owners",
        "result.current_q_a_state_owners",
        "result.current_dsa_query_owners",
        "result.current_dsa_head_weights_owners",
    ]
    with pytest.raises(BenchmarkValidationError, match="boundary shapes drifted"):
        validate_feature2_main_stablehlo(
            _main_stablehlo(full_width_rounded_then_slice=True),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


def test_feature2_main_position113_stablehlo_pins_named_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stablehlo = _main_stablehlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    )
    _pin_synthetic_position113_stablehlo(monkeypatch, stablehlo)
    report = validate_feature2_main_stablehlo(
        stablehlo,
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
        source_jaxpr_sha256=(
            "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
        ),
    )
    assert report["output_count"] == 24
    assert report["observe_position_113"] is True
    assert report["result_infos"][15:] == [
        "result.position113_normalized_hidden_owners",
        "result.position113_q_a_state_owners",
        "result.position113_dsa_query_owners",
        "result.position113_dsa_head_weights_owners",
        "result.position113_current_key_owners",
        "result.position113_selected_positions_owners",
        "result.position113_selected_valid_counts_owners",
        "result.position113_selected_scores_owners",
        "result.position113_observation_count_owners",
    ]
    with pytest.raises(BenchmarkValidationError, match="causal source JAXpr"):
        validate_feature2_main_stablehlo(
            stablehlo,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            source_jaxpr_sha256="0" * 64,
        )
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(
            stablehlo.replace(
                "result.position113_current_key_owners",
                "result.position113_selected_scores_owners",
                1,
            ),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            source_jaxpr_sha256=(
                "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
            ),
        )


def _db518_acquisition_stablehlo() -> str:
    return _main_stablehlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
    ) + (
        "\n// acquisition-only DB518 chunk geometry: "
        "tensor<2048x3072xbf16> tensor<2012x3072xbf16> "
        "tensor<2048x6144xbf16> tensor<2012x6144xbf16> "
        "tensor<2048x128xf32> tensor<2012x128xf32>\n"
    )


def test_feature2_db518_stablehlo_requires_exact_source_and_chunk_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stablehlo = _db518_acquisition_stablehlo()
    monkeypatch.setattr(
        feature2_hlo_module,
        "FEATURE2_DB518_STABLEHLO_SHA256",
        sha256(stablehlo.encode()).hexdigest(),
    )
    source_sha = "f87c0f162523bf6221d2824aa4b7268f1c770a55d2f8cde6da7506f595e9447f"
    report = validate_feature2_main_stablehlo(
        stablehlo,
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
        observe_position_113=True,
        exact_layer0_prompt_keys=True,
        source_jaxpr_sha256=source_sha,
    )
    assert report["passed"] is True
    assert report["exact_layer0_prompt_keys"] is True
    assert report["identity_acquisition_only"] is False
    assert report["source_jaxpr_sha256"] == source_sha
    with pytest.raises(BenchmarkValidationError, match="exact causal source JAXpr"):
        validate_feature2_main_stablehlo(
            stablehlo,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            exact_layer0_prompt_keys=True,
            source_jaxpr_sha256=(
                "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
            ),
        )
    with pytest.raises(BenchmarkValidationError, match="boundary shapes drifted"):
        validate_feature2_main_stablehlo(
            stablehlo.replace("tensor<2012x128xf32>", "tensor<2011x128xf32>"),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
            observe_position_113=True,
            exact_layer0_prompt_keys=True,
            source_jaxpr_sha256=source_sha,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "full_width_rounded_then_slice": False,
            "sealed_boundary_capture": True,
            "observe_position_113": True,
        },
        {
            "full_width_rounded_then_slice": True,
            "sealed_boundary_capture": False,
            "observe_position_113": True,
        },
        {
            "full_width_rounded_then_slice": True,
            "sealed_boundary_capture": True,
            "observe_position_113": False,
        },
    ),
)
def test_feature2_db518_hlo_requires_sealed_full_width_observer(kwargs) -> None:
    with pytest.raises(ValueError, match="DB518 layer-0 acquisition requires"):
        validate_feature2_main_stablehlo(
            _db518_acquisition_stablehlo(),
            exact_layer0_prompt_keys=True,
            source_jaxpr_sha256=(
                "f87c0f162523bf6221d2824aa4b7268f1c770a55d2f8cde6da7506f595e9447f"
            ),
            **kwargs,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            "result.current_normalized_hidden_owners",
            "result.current_q_a_state_owners",
            1,
        ),
        lambda value: value.replace("%result#9, %result#10", "%result#10", 1),
    ),
)
def test_feature2_main_successor_stablehlo_refuses_decoy_or_unbound_results(
    mutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _main_stablehlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    _pin_synthetic_sealed_stablehlo(monkeypatch, original)
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(
            mutation(original),
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


def test_feature2_main_successor_stablehlo_refuses_module_wide_shape_decoys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _main_stablehlo(full_width_rounded_then_slice=True)
    sealed = _main_stablehlo(
        full_width_rounded_then_slice=True,
        sealed_boundary_capture=True,
    )
    _pin_synthetic_sealed_stablehlo(monkeypatch, sealed)
    decoy = (
        historical + "\n// tensor<2x1x6144xbf16> tensor<2x1x2048xbf16> "
        "tensor<2x1x32x128xf32> tensor<2x1x32xf32>\n"
    )
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(
            decoy,
            full_width_rounded_then_slice=True,
            sealed_boundary_capture=True,
        )


def test_feature2_main_successor_stablehlo_refuses_hybrid_or_delayed_slice() -> None:
    original = _main_stablehlo(full_width_rounded_then_slice=True)
    mutations = (
        original.replace(
            "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
            "greenfield_fp8_strategy_nd_o_m8_k512_n3072",
            1,
        ),
        original.replace(
            "%ah1_0_0 = stablehlo.slice",
            "%ah1_0_0 = stablehlo.copy",
            1,
        ),
        original + "tensor<16x1x6144xbf16> full_leaf_stack",
    )
    for mutation in mutations:
        with pytest.raises(BenchmarkValidationError):
            validate_feature2_main_stablehlo(
                mutation, full_width_rounded_then_slice=True
            )


def _swap_first_successor_attention_half_ranges(stablehlo: str) -> str:
    lower = "%ah0_0_0 = stablehlo.slice %ar0_0 [0:1, 0:3072]"
    upper = "%ah1_0_0 = stablehlo.slice %ar0_0 [0:1, 3072:6144]"
    sentinel = "%ah0_0_0 = stablehlo.slice %ar0_0 [0:1, 999:999]"
    assert lower in stablehlo and upper in stablehlo and sentinel not in stablehlo
    return (
        stablehlo.replace(lower, sentinel, 1)
        .replace(
            upper,
            "%ah1_0_0 = stablehlo.slice %ar0_0 [0:1, 0:3072]",
            1,
        )
        .replace(
            sentinel,
            "%ah0_0_0 = stablehlo.slice %ar0_0 [0:1, 3072:6144]",
            1,
        )
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            "%ar0_0 = stablehlo.slice %a0_0 [0:1, 0:6144]",
            "%ar0_0 = stablehlo.slice %a0_0 [1:2, 0:6144]",
            1,
        ),
        lambda value: value.replace(
            "%ah1_0_0 = stablehlo.slice %ar0_0 [0:1, 3072:6144]",
            "%ah1_0_0 = stablehlo.slice %ar0_0 [0:1, 0:3072]",
            1,
        ),
        _swap_first_successor_attention_half_ranges,
        lambda value: value.replace(
            "%dh1_0_0 = stablehlo.slice %dr0_0 [0:1, 3072:6144]",
            "%dh1_0_0 = stablehlo.slice %dr0_0 [0:1, 0:3072]",
            1,
        ),
        lambda value: value.replace(
            "%ah0_0_0 = stablehlo.slice %ar0_0 [0:1, 0:3072]",
            "%ah0_0_0 = stablehlo.slice %ar0_0 [0:1:1, 0:6144:2]",
            1,
        ),
    ),
)
def test_feature2_successor_stablehlo_refuses_slice_geometry_or_owner_drift(
    mutation,
) -> None:
    original = _main_stablehlo(full_width_rounded_then_slice=True)
    attacked = mutation(original)
    assert attacked != original
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(attacked, full_width_rounded_then_slice=True)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions = 2", "num_partitions = 32"),
        lambda value: value.replace("[[0, 1]]", "[[0, 1, 2, 3]]"),
        lambda value: value.replace("[[0, 1], [1, 0]]", "[[0, 2], [2, 0]]"),
        lambda value: value + "tensor<32x6144xbf16> outside_compilation",
    ),
)
def test_feature2_main_stablehlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(mutation(_main_stablehlo()))
