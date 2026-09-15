"""The native path's shared helpers moved verbatim out of the campaign modules.

BudgetedCalls, memory-owner validation, voted traces, evidence replay, the
compile-with-originals helper, the authenticated inventory reader, the WK
program builder and the canonical-dense source pins now live in native-owned
modules. Every moved definition must equal its b667f00f original by AST, and
importing the native modules must not load the retired campaign hubs.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"
PROBE = "scripts/greenfield/probe_ws32_prefill_layer.py"
WINDOW_WORKER = "scripts/greenfield/prefill_window_worker.py"
MOVES = {
    "scripts/greenfield/ws32_budgeted_calls.py": (
        (WINDOW_WORKER, "save_arrays"),
        (WINDOW_WORKER, "validate_memory_owners"),
        (WINDOW_WORKER, "BudgetedCalls"),
        ("scripts/greenfield/prefill_phase_baseline.py", "start_device_trace"),
        ("scripts/greenfield/prefill_phase_baseline.py", "voted_trace"),
        ("scripts/greenfield/prefill_window_evidence.py", "same_json"),
        ("scripts/greenfield/prefill_window_evidence.py", "validate_call_sequence"),
    ),
    "scripts/greenfield/ws32_compile_originals.py": (
        (PROBE, "_write_compiler_original"),
        (PROBE, "compile_program"),
        (PROBE, "authenticated_inventory"),
        (PROBE, "build_wk_programs"),
    ),
    "scripts/greenfield/ws32_dense_canonical_source.py": (
        ("scripts/greenfield/ws32_dense_canonical.py", "MODEL_SOURCE_OVERRIDES"),
        ("scripts/greenfield/ws32_dense_canonical.py", "require_source"),
    ),
    "scripts/greenfield/prefill_window_admission.py": (
        ("scripts/greenfield/prefill_layer_hlo.py", "FEATURE"),
        ("scripts/greenfield/prefill_layer_hlo.py", "EXPERT"),
        ("scripts/greenfield/prefill_window_admission.py", "PROGRAMS"),
        ("scripts/greenfield/prefill_window_admission.py", "REQUIRED_RESERVE_BYTES"),
        ("scripts/greenfield/prefill_window_admission.py", "registered_programs"),
        ("scripts/greenfield/prefill_window_admission.py", "expected_collectives"),
        ("scripts/greenfield/prefill_window_admission.py", "memory_budget"),
    ),
}
HUBS = (
    "prefill_layer_hlo",
    "prefill_moe_precision_hlo",
    "prefill_window_locations",
    "probe_ws32_prefill_layer",
    "prefill_window_worker",
    "prefill_phase_baseline",
    "prefill_window_evidence",
    "ws32_dense_canonical",
    "ws32_history_preflight",
    "ws32_dense_frontier_worker",
)
NATIVE_IMPORTERS = (
    "scripts/greenfield/ws32_native_benchmark_worker.py",
    "scripts/greenfield/ws32_native_benchmark_memory.py",
    "scripts/greenfield/ws32_native_benchmark_observability.py",
    "scripts/greenfield/ws32_native_benchmark_evidence.py",
    "scripts/greenfield/ws32_delivery_decode.py",
    "scripts/greenfield/ws32_delivery_wk.py",
    "scripts/greenfield/ws32_delivery_phase_evidence.py",
    "scripts/greenfield/ws32_history_call_evidence.py",
    "scripts/greenfield/ws32_history_worker_storage.py",
    "scripts/greenfield/ws32_phase_weights.py",
    "scripts/greenfield/ws32_rolled_prefill_compile.py",
    "scripts/greenfield/ws32_dense_frontier_admission.py",
)


def _definition(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise AssertionError(name)


def _baseline(path: str) -> ast.Module:
    return ast.parse(
        subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO, text=True)
    )


def _current(path: str) -> ast.Module:
    return ast.parse((REPO / path).read_text())


def _without_annotations(node: ast.AST) -> ast.AST:
    class Strip(ast.NodeTransformer):
        def visit_arg(self, arg):
            arg.annotation = None
            return arg

    return Strip().visit(ast.parse(ast.unparse(node)))


@pytest.mark.parametrize(
    ("module", "origin", "name"),
    [(m, o, n) for m, moves in MOVES.items() for o, n in moves],
)
def test_moved_definition_matches_baseline_ast(module, origin, name):
    moved = _definition(_current(module), name)
    original = _definition(_baseline(origin), name)
    if name == "BudgetedCalls":
        assert ast.dump(_without_annotations(moved)) == ast.dump(
            _without_annotations(original)
        )
        init = next(n for n in moved.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        journal = next(a for a in init.args.kwonlyargs if a.arg == "journal")
        assert ast.unparse(journal.annotation) == "Ws32NumericalJournal"
        return
    assert ast.dump(moved) == ast.dump(original)


def test_wk_admission_program_constants_match_the_retired_worker():
    worker = _baseline("scripts/greenfield/ws32_dense_frontier_worker.py")
    namespace: dict = {}
    for name in ("GRAPH", "PROGRAMS"):
        exec(ast.unparse(_definition(worker, name)), namespace)
    admission = _current("scripts/greenfield/ws32_dense_frontier_admission.py")
    current: dict = {}
    for name in ("GRAPH", "PROGRAMS"):
        exec(ast.unparse(_definition(admission, name)), current)
    assert (current["GRAPH"], current["PROGRAMS"]) == (namespace["GRAPH"], namespace["PROGRAMS"])
    assert current["PROGRAMS"] == ("wk_decode", "wk_promote", "dense01")


def test_history_storage_uses_the_identical_stdlib_path_guard():
    ours = _definition(_current("glm_tpu/host_paths.py"), "_plain_path")
    theirs = _definition(
        _baseline("scripts/greenfield/ws32_history_preflight.py"), "_plain_path"
    )
    assert ast.dump(ours) == ast.dump(theirs)


def test_native_importers_no_longer_import_any_campaign_hub():
    for path in NATIVE_IMPORTERS:
        for node in ast.walk(_current(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.rsplit(".", 1)[-1] not in HUBS, (path, node.module)
                for alias in node.names:
                    assert alias.name not in HUBS, (path, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.rsplit(".", 1)[-1] not in HUBS, (path, alias.name)


def test_native_helper_modules_do_not_load_campaign_hubs():
    code = """
import json, sys
import scripts.greenfield.ws32_budgeted_calls
import scripts.greenfield.ws32_compile_originals
import scripts.greenfield.ws32_dense_canonical_source
print(json.dumps(sorted(m.rsplit('.', 1)[-1] for m in sys.modules if m.startswith('scripts.greenfield.'))))
"""
    import json

    loaded = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=REPO,
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            text=True,
            timeout=60,
        )
    )
    assert not set(loaded) & set(HUBS), sorted(set(loaded) & set(HUBS))
    assert "ws32_budgeted_calls" in loaded and "ws32_compile_originals" in loaded
