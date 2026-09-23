"""S2a: the worker's helpers live in production modules, moved verbatim.

Every moved definition equals its 181c013e original by AST; the two file writers lose only their
research-namespace branches, which never match the worker's HLO directory. The research homes
re-export the moved objects for their remaining importers (and keep their own research writers),
and the worker and runtime modules import nothing from ``scripts.greenfield``.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
BASE = "181c013e84ac7a2d1c2069feaa9e8aa90da51af5"
RUN_SHORT = "scripts/greenfield/run_short_decoder_ws32.py"
ORIGINALS = "scripts/greenfield/ws32_compile_originals.py"
MICROBENCH = "scripts/greenfield/microbench_fp8_matmul.py"
NATIVE_PROGRAMS = "scripts/greenfield/ws32_native_benchmark_programs.py"
ONE_LAYER = "glm_tpu/greenfield/benchmarking/ws32_one_layer.py"
STAGE_LOCAL = "glm_tpu/greenfield/kernels/stage_local.py"
# (new home, 181c013e home, name): moved without any change
VERBATIM = (
    ("glm_tpu/distributed/parallel_state.py", RUN_SHORT, "_initialize_runtime"),
    ("glm_tpu/distributed/parallel_state.py", RUN_SHORT, "_batched_fleet_all"),
    ("glm_tpu/optimized/topology_binding.py", RUN_SHORT, "_device_record"),
    ("glm_tpu/optimized/topology_binding.py", ONE_LAYER, "_TOPOLOGY_CAPTURE_KEYS"),
    ("glm_tpu/optimized/topology_binding.py", ONE_LAYER, "validate_ws32_topology_fleet"),
    ("glm_tpu/runner/compilation_manager.py", ORIGINALS, "compile_program"),
    ("glm_tpu/runner/compilation_manager.py", MICROBENCH, "_compiled_memory"),
    ("glm_tpu/runner/compilation_manager.py", MICROBENCH, "_memory_stats"),
    ("glm_tpu/runner/kv_cache_manager.py", NATIVE_PROGRAMS, "build_cache_initializer"),
    ("glm_tpu/optimized/bf16_resident.py", ORIGINALS, "build_wk_programs"),
    ("glm_tpu/greenfield/partitioning/source_inventory.py", ORIGINALS, "authenticated_inventory"),
    ("glm_tpu/greenfield/kernels/prefill_cache.py", STAGE_LOCAL, "_require_decode_metadata"),
)
# (research home, module attribute, production module, attribute): the same object
REEXPORTS = (
    ("scripts.greenfield.run_short_decoder_ws32", "_initialize_runtime",
     "glm_tpu.distributed.parallel_state", "_initialize_runtime"),
    ("scripts.greenfield.run_short_decoder_ws32", "_batched_fleet_all",
     "glm_tpu.distributed.parallel_state", "_batched_fleet_all"),
    ("glm_tpu.greenfield.benchmarking.ws32_one_layer", "validate_ws32_topology_fleet",
     "glm_tpu.optimized.topology_binding", "validate_ws32_topology_fleet"),
    ("glm_tpu.greenfield.benchmarking", "validate_ws32_topology_fleet",
     "glm_tpu.optimized.topology_binding", "validate_ws32_topology_fleet"),
    ("scripts.greenfield.ws32_compile_originals", "authenticated_inventory",
     "glm_tpu.greenfield.partitioning.source_inventory", "authenticated_inventory"),
    ("scripts.greenfield.ws32_compile_originals", "build_wk_programs",
     "glm_tpu.optimized.bf16_resident", "build_wk_programs"),
    ("scripts.greenfield.ws32_native_benchmark_programs", "build_cache_initializer",
     "glm_tpu.runner.kv_cache_manager", "build_cache_initializer"),
    ("scripts.greenfield.microbench_fp8_matmul", "_compiled_memory",
     "glm_tpu.runner.compilation_manager", "_compiled_memory"),
    ("scripts.greenfield.microbench_fp8_matmul", "_memory_stats",
     "glm_tpu.runner.compilation_manager", "_memory_stats"),
    ("glm_tpu.greenfield.kernels.stage_local", "_require_decode_metadata",
     "glm_tpu.greenfield.kernels.prefill_cache", "_require_decode_metadata"),
)


def _definition(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise AssertionError(name)


def _baseline(path: str) -> ast.Module:
    return ast.parse(subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO, text=True))


def _current(path: str) -> ast.Module:
    return ast.parse((REPO / path).read_text())


@pytest.mark.parametrize(("home", "origin", "name"), VERBATIM)
def test_moved_definition_equals_its_181c013e_original(home, origin, name):
    assert ast.dump(_definition(_current(home), name)) == ast.dump(_definition(_baseline(origin), name))


def test_production_writers_keep_only_the_original_default_branch():
    # _write_compiler_original: the original ends in `if <history protocol>: ... else: write_text`;
    # the production body is that else branch (after its own docstring).
    original = _definition(_baseline(ORIGINALS), "_write_compiler_original")
    production = _definition(_current("glm_tpu/runner/compilation_manager.py"), "_write_compiler_original")
    assert ast.dump(original.args) == ast.dump(production.args)
    final = original.body[-1]
    assert isinstance(final, ast.If) and len(final.orelse) == 1
    assert [ast.dump(n) for n in production.body[1:]] == [ast.dump(n) for n in final.orelse]
    # _atomic_json: three research-namespace `if` blocks, then the plain atomic write.
    original = _definition(_baseline(MICROBENCH), "_atomic_json")
    production = _definition(_current("glm_tpu/runner/compilation_manager.py"), "_atomic_json")
    assert ast.dump(original.args) == ast.dump(production.args)
    assert all(isinstance(n, ast.If) for n in original.body[:3])
    assert [ast.dump(n) for n in production.body] == [ast.dump(n) for n in original.body[3:]]


@pytest.mark.parametrize(("path", "name"), [(ORIGINALS, "_write_compiler_original"), (ORIGINALS, "compile_program"),
                                            (MICROBENCH, "_atomic_json")])
def test_research_writers_keep_their_namespace_branches(path, name):
    assert ast.dump(_definition(_current(path), name)) == ast.dump(_definition(_baseline(path), name))


def test_research_homes_reexport_the_moved_objects():
    import importlib

    for old, old_name, new, new_name in REEXPORTS:
        assert getattr(importlib.import_module(old), old_name) is getattr(importlib.import_module(new), new_name), old


def test_worker_and_runtime_modules_import_nothing_from_the_research_layer():
    code = """
import json, sys
for name in ("scripts.release.ws32_optimized_worker", "glm_tpu.distributed.parallel_state",
             "glm_tpu.optimized.runtime", "glm_tpu.optimized.batched_runtime", "glm_tpu.optimized.topology_binding",
             "glm_tpu.runner.compilation_manager", "glm_tpu.runner.kv_cache_manager",
             "glm_tpu.greenfield.partitioning.source_inventory"):
    __import__(name)
print(json.dumps(sorted(sys.modules)))
"""
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=REPO,
                                                env=dict(os.environ, JAX_PLATFORMS="cpu"), text=True, timeout=120))
    research = [m for m in loaded if m.startswith("scripts.greenfield")
                or m.startswith(("glm_tpu.greenfield.benchmarking", "glm_tpu.greenfield.validation"))]
    assert research == []


def test_production_sources_name_no_research_home():
    paths = [*(path for package in ("optimized", "runner", "distributed")
               for path in sorted((REPO / "glm_tpu" / package).glob("*.py"))),
             REPO / "scripts/release/ws32_optimized_worker.py",
             REPO / "scripts/release/launch_ws32_optimized_request.py"]
    for path in paths:
        package = path.relative_to(REPO).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = list(package[:len(package) - node.level + 1]) if node.level else []
                names = [".".join([*base, *([node.module] if node.module else [])])]
            for name in names:
                assert not name.startswith(("scripts.greenfield", "glm_tpu.greenfield.benchmarking",
                                            "glm_tpu.greenfield.validation")), (path.name, name)
