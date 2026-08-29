from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

WRAPPER = Path("scripts/greenfield/run_pp16_feature2_prefill_acquisition.sh")


def _python_heredoc_after(source: str, marker: str) -> str:
    section = source.split(marker, 1)[1]
    return section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def _all_python_heredocs(source: str) -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", source, flags=re.DOTALL)


def test_feature2_acquisition_wrapper_is_default_off_and_serialized() -> None:
    source = WRAPPER.read_text()
    for marker in (
        "GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE:-0",
        "GLM_GREENFIELD_PP16_FEATURE2_MODE:-off",
        "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE:-0",
        "GLM_GREENFIELD_PP16_FEATURE2_OBSERVE_POSITION_113:-0",
        "compile_only",
        "/home/gianl/glm-run/.glm_pod_workload.lock",
        "/home/gianl/.glm-tpu-rsync.lock",
        "flock -n 9",
        "flock 8",
        "strict_census pre",
        "strict_census post",
        "ray_enum=",
        "--worker=all",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "--compile-only 1",
        "--full-width-rounded-then-slice",
        "--observe-position-113",
        "full_width_rounded_then_slice",
        "sealed_boundary_capture",
        "expected_terminal_shapes",
        "expected_terminal_dtypes",
        "expected_stable_types",
        "expected_optimized_roots",
        "expected_sealed_bindings",
        "expected_observer_root_hints",
        "expected_optimized_observer_bindings",
        "sealed_acquisition_root_hints",
        "position113_observer_acquisition_root_hints",
        "position113_observer_root_hints_causal",
        "expected_jaxpr_sha",
        "expected_raw_jaxpr_sha",
        "jaxpr_canonicalizer_version",
        "jaxpr_runtime_mesh",
        "jaxpr_runtime_mesh_fragment_count",
        "expected_causal_contract",
        "source_jaxpr_sha256",
        "expected_stable_sha",
        "expected_canonical_sha",
        "expected_canonical_bytes",
        "expected_canonical_stack_refs",
        "sealed_canonical_hlo_identity",
        "validate_feature2_sealed_hlo_archive_identity",
        "POSITION113_MAIN_STABLEHLO_SHA",
        "POSITION113_MAIN_CANONICAL_HLO_SHA",
        "POSITION113_MAIN_CANONICAL_HLO_BYTES",
        "POSITION113_MAIN_CANONICAL_STACK_REFS",
        "archive_identity['stablehlo_sha256']",
        "archive_identity['optimized_hlo_sha256']",
        "recomputed_canonical",
        "canonicalizer_code_hash",
        "raw HLO identity records drifted",
        "canonical-HLO artifact drifted",
        "StableHLO sealed terminal drifted",
        "optimized-HLO sealed terminal drifted",
        "abstract sealed terminal drifted",
    ):
        assert marker in source


def test_feature2_acquisition_postrun_verifier_import_is_in_exact_scope() -> None:
    source = WRAPPER.read_text()
    source_auth = _python_heredoc_after(
        source,
        'say "authenticating the exact selected runtime and accepted event-1 lineage"',
    )
    verifier = _python_heredoc_after(
        source,
        'say "recomputing every HLO/source/load claim without JAX"',
    )

    assert "validate_feature2_sealed_hlo_archive_identity" not in source_auth
    compile(verifier, "<pp16-feature2-postrun-verifier>", "exec")
    tree = ast.parse(verifier)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "glm_tpu.greenfield.benchmarking.pp16_feature2_hlo"
        and any(
            alias.name == "validate_feature2_sealed_hlo_archive_identity"
            for alias in node.names
        )
        for node in tree.body
    )
    assert "validate_feature2_sealed_hlo_archive_identity(" in verifier


def test_feature2_acquisition_compiles_every_embedded_python_program() -> None:
    programs = _all_python_heredocs(WRAPPER.read_text())
    assert len(programs) == 3
    for index, program in enumerate(programs):
        compile(program, f"<pp16-feature2-heredoc-{index}>", "exec")


def test_feature2_acquisition_wrapper_refuses_the_rejected_variant() -> None:
    source = WRAPPER.read_text()
    assert "[[ $FULL_WIDTH_ROUNDED_THEN_SLICE == 1 ]]" in source
    assert "runner_variant_args=(--full-width-rounded-then-slice)" in source
    assert "runner_variant_args+=(--observe-position-113)" in source
    environment = os.environ.copy()
    environment.update(
        {
            "GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE": "1",
            "GLM_GREENFIELD_PP16_FEATURE2_MODE": "compile_only",
            "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "admitted successor" in completed.stderr


def test_feature2_acquisition_wrapper_refuses_invalid_observer_flag_early() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "GLM_GREENFIELD_PP16_FEATURE2_ACQUIRE": "1",
            "GLM_GREENFIELD_PP16_FEATURE2_MODE": "compile_only",
            "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE": "1",
            "GLM_GREENFIELD_PP16_FEATURE2_OBSERVE_POSITION_113": "2",
        }
    )
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "observer flag must be 0 or 1" in completed.stderr


def test_feature2_acquisition_wrapper_has_no_numerical_or_db_success_path() -> None:
    source = WRAPPER.read_text()
    assert "'main_executed':False" in source
    assert "'numerical_claim':False" in source
    assert "'performance_claim':False" in source
    assert "HLO_ACQUIRED" in source
    assert "results.db" not in source
    assert "/SUCCESS" not in source
    assert "--warmup" not in source
    assert "--iterations" not in source


def test_feature2_acquisition_wrapper_pins_region_sources_and_remote_hashes() -> None:
    source = WRAPPER.read_text()
    for marker in (
        "APPROVED_LOCATION=US-CENTRAL2",
        "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab",
        "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2",
        "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da",
        "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054",
        "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d",
        "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4",
        "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5",
        "e514fc28e9d8c30bc7de9d70f01ae04002666494446e2f4a06a7fe6c66901e65",
        "a2ef16a7a55876099124d0ac4bd139f86c6318b27c0e48fef5d64193ed0a3023",
        'gcloud storage cat "$REMOTE_PREFIX/$relative"',
        "upload_failure_diagnostic",
        "verify_failure_diagnostic",
        "verify_success_evidence",
        "DIAGNOSTIC_UPLOAD_FAILED after two attempts",
        'gcloud storage ls --recursive "$REMOTE_PREFIX/**"',
        "--no-clobber",
    ):
        assert marker in source


def test_feature2_acquisition_wrapper_is_bash_syntax_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
