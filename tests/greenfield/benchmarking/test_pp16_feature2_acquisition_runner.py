from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


RUNNER = Path("scripts/greenfield/acquire_pp16_feature2_prefill.py")


def _calls(source: str, name: str) -> int:
    tree = ast.parse(source)
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def test_feature2_acquisition_runner_compiles_but_never_executes_main() -> None:
    source = RUNNER.read_text()
    assert _calls(source, "main_compiled") == 0
    assert _calls(source, "query_compiled") == 1
    assert _calls(source, "wk_decode_compiled") == 1
    assert _calls(source, "wk_promote_compiled") == 1
    assert '"main_executed": False' in source
    assert '"numerical_claim": False' in source
    assert '"performance_claim": False' in source
    assert '"status": "HLO_ACQUIRED"' in source


def test_feature2_acquisition_runner_requires_all_fail_closed_contracts() -> None:
    source = RUNNER.read_text()
    for marker in (
        "inspect_feature2_event1_lineage(",
        "inspect_feature2_selective_plan(",
        "load_feature2_selective_checkpoint(",
        "pack_dense_final_layout=True",
        "validate_feature2_materializer_optimized_hlo(",
        "validate_feature2_prefill_jaxpr(",
        "validate_feature2_prefill_result_abstract(",
        "validate_feature2_main_stablehlo(",
        "validate_feature2_main_optimized_hlo(",
        "jax.clear_caches()",
    ):
        assert marker in source


def test_feature2_acquisition_persists_stablehlo_before_compile() -> None:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    compile_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compile"
    )
    calls = [
        node
        for node in ast.walk(compile_function)
        if isinstance(node, ast.Call)
    ]
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
    main_atomic = source.index(
        '_atomic_text(\n            args.hlo_dir / "feature2_main.stablehlo.mlir"'
    )
    main_validate = source.index("validate_feature2_main_stablehlo(main_stablehlo)")
    main_compile = source.index("main_lowered.compile()")
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
    source = RUNNER.read_text()
    assert 'choices=(1,)' in source


def test_feature2_acquisition_runner_is_syntax_valid() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
