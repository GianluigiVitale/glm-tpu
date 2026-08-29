from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

RUNNER = Path("scripts/greenfield/acquire_pp16_feature2_prefill.py")


def _calls(source: str, name: str) -> int:
    tree = ast.parse(source)
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def test_feature2_acquisition_entrypoint_compiles_but_never_executes_main() -> None:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    main_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "main_compiled"
    )
    ancestor = parents[main_call]
    while not isinstance(ancestor, ast.If):
        ancestor = parents[ancestor]
    assert isinstance(ancestor.test, ast.Name)
    assert ancestor.test.id == "execute_main"
    assert _calls(source, "main_compiled") == 1
    assert _calls(source, "query_compiled") == 1
    assert _calls(source, "wk_decode_compiled") == 1
    assert _calls(source, "wk_promote_compiled") == 1
    assert "return run_feature2(parse_args(), execute_main=False)" in source
    assert '"main_executed": execute_main' in source
    assert '"sealed_boundary_capture": program.sealed_boundary_capture' in source
    assert '"observe_position_113": program.observe_position_113' in source
    assert '"exact_layer0_prompt_keys": program.exact_layer0_prompt_keys' in source
    assert '"numerical_claim": False' in source
    assert '"performance_claim": False' in source
    assert '"NUMERICAL_CAPTURED" if execute_main else "HLO_ACQUIRED"' in source


def test_feature2_acquisition_runner_requires_all_fail_closed_contracts() -> None:
    source = RUNNER.read_text()
    for marker in (
        "inspect_feature2_event1_lineage(",
        "inspect_feature2_selective_plan(",
        "load_feature2_selective_checkpoint(",
        "pack_dense_final_layout=True",
        "full_width_rounded_then_slice=(",
        "observe_position_113=observe_position_113",
        "exact_layer0_prompt_keys=exact_layer0_prompt_keys",
        "validate_feature2_materializer_optimized_hlo(",
        "validate_feature2_prefill_jaxpr(",
        "validate_feature2_prefill_result_abstract(",
        "validate_feature2_main_stablehlo(",
        "validate_feature2_main_optimized_hlo(",
        "jax.clear_caches()",
    ):
        assert marker in source
    tree = ast.parse(source)
    for name in (
        "validate_feature2_main_stablehlo",
        "validate_feature2_main_optimized_hlo",
    ):
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )
        keywords = {item.arg: item.value for item in call.keywords}
        assert "full_width_rounded_then_slice" in keywords
        assert (
            ast.unparse(keywords["full_width_rounded_then_slice"])
            == "program.full_width_rounded_then_slice"
        )
        assert "sealed_boundary_capture" in keywords
        assert (
            ast.unparse(keywords["sealed_boundary_capture"])
            == "program.sealed_boundary_capture"
        )
        assert "observe_position_113" in keywords
        assert (
            ast.unparse(keywords["observe_position_113"])
            == "program.observe_position_113"
        )
        assert "exact_layer0_prompt_keys" in keywords
        assert (
            ast.unparse(keywords["exact_layer0_prompt_keys"])
            == "program.exact_layer0_prompt_keys"
        )
        if name == "validate_feature2_main_stablehlo":
            assert "source_jaxpr_sha256" in keywords
            assert (
                ast.unparse(keywords["source_jaxpr_sha256"])
                == "jaxpr_contract['jaxpr_sha256']"
            )


def test_feature2_acquisition_persists_stablehlo_before_compile() -> None:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    compile_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compile"
    )
    calls = [node for node in ast.walk(compile_function) if isinstance(node, ast.Call)]
    atomic_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_atomic_text"
    )
    compile_line = min(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
    )
    assert atomic_line < compile_line
    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_feature2"
    )
    run_calls = [node for node in ast.walk(run_function) if isinstance(node, ast.Call)]
    main_atomic = next(
        node.lineno
        for node in run_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_atomic_text"
        and "feature2_main.stablehlo.mlir" in ast.unparse(node)
    )
    main_validate = next(
        node.lineno
        for node in run_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "validate_feature2_main_stablehlo"
    )
    main_compile = next(
        node.lineno
        for node in run_calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and ast.unparse(node.func.value) == "main_lowered"
    )
    assert main_atomic < main_validate < main_compile


def test_feature2_acquisition_runner_cli_is_compile_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--compile-only" in completed.stdout
    assert "--full-width-rounded-then-slice" in completed.stdout
    assert "--observe-position-113" in completed.stdout
    assert "--exact-layer0-prompt-keys" in completed.stdout
    source = RUNNER.read_text()
    assert "choices=(1,)" in source


def test_feature2_db518_execution_is_exposed_only_by_the_pinned_entrypoint() -> None:
    source = RUNNER.read_text()
    numerical = Path("scripts/greenfield/execute_pp16_feature2_prefill.py").read_text()
    assert "return run_feature2(parse_args(), execute_main=False)" in source
    assert "compile-only until acquired HLO is pinned" not in source
    assert 'parser.add_argument("--exact-layer0-prompt-keys"' in numerical
    for pin in (
        "--expected-main-stablehlo-sha256",
        "--expected-main-canonical-hlo-sha256",
        "--expected-main-canonical-hlo-byte-count",
        "--expected-main-stack-frame-reference-count",
    ):
        assert pin in numerical
    assert (
        "return run_feature2(_require_exact_db518_pins(parse_args()), execute_main=True)"
        in numerical
    )


def test_feature2_db518_direct_execution_rejects_caller_chosen_hlo_pin(
    tmp_path: Path,
) -> None:
    runner = Path("scripts/greenfield/execute_pp16_feature2_prefill.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--expected-code-hash",
            "unused-before-pin-validation",
            "--runtime-root",
            str(tmp_path),
            "--token-oracle-dir",
            str(tmp_path),
            "--dsa-oracle-dir",
            str(tmp_path),
            "--layer1-internal-reference",
            str(tmp_path / "layer1.npz"),
            "--db529-internal-dir",
            str(tmp_path),
            "--full-width-rounded-then-slice",
            "--observe-position-113",
            "--exact-layer0-prompt-keys",
            "--expected-main-stablehlo-sha256",
            "caller-chosen",
            "--expected-main-canonical-hlo-sha256",
            "56b9b88dc081dd5d3ea2219d46be641128cbed6e9759754be7f87578b51b5d8c",
            "--expected-main-canonical-hlo-byte-count",
            "6863602",
            "--expected-main-stack-frame-reference-count",
            "15339",
            "--expected-jax-version",
            "0.10.1",
            "--expected-jaxlib-version",
            "0.10.1",
            "--expected-libtpu-version",
            "0.0.41",
            "--output",
            str(tmp_path / "summary.json"),
            "--hlo-dir",
            str(tmp_path / "hlo"),
            "--result-npz",
            str(tmp_path / "result.npz"),
        ],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "exact DB518 acquisition pin drift" in completed.stderr


def test_feature2_acquisition_runner_is_syntax_valid() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
