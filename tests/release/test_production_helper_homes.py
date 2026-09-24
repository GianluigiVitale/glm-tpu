"""S2a/S2f: the production helpers live in production modules, moved verbatim.

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
    ("glm_tpu/optimized/source_inventory.py", ORIGINALS, "authenticated_inventory"),
    ("glm_tpu/optimized/prefill_cache.py", STAGE_LOCAL, "_require_decode_metadata"),
)


# S2f: the production definitions of the research package moved into glm_tpu.optimized, verbatim.
# Whole files (every top-level definition of the S2f base file) ...
S2F_BASE = "c8ee4fff5003d74cd8a75b68135a8f4bf7951280"
S2F_WHOLE = (
    ("glm_tpu/optimized/errors.py", "glm_tpu/greenfield/errors.py"),
    ("glm_tpu/optimized/geometry.py", "glm_tpu/greenfield/types.py"),
    ("glm_tpu/optimized/mesh.py", "glm_tpu/greenfield/sharding/ws32.py"),
    ("glm_tpu/optimized/hlo_contract.py", "glm_tpu/greenfield/sharding/hlo_contract.py"),
    ("glm_tpu/optimized/source_inventory.py", "glm_tpu/greenfield/partitioning/source_inventory.py"),
    ("glm_tpu/optimized/checkpoint_placement.py", "glm_tpu/greenfield/checkpoint/ws32_runtime.py"),
    ("glm_tpu/optimized/runtime_checkpoint.py", "glm_tpu/greenfield/checkpoint/ws32_runtime_checkpoint.py"),
    ("glm_tpu/optimized/prefill_panel_fp8.py", "glm_tpu/greenfield/kernels/pallas/prefill_panel_fp8.py"),
    ("glm_tpu/optimized/prefill_cache.py", "glm_tpu/greenfield/kernels/prefill_cache.py"),
    ("glm_tpu/optimized/prefill_expert_panels.py", "glm_tpu/greenfield/kernels/prefill_expert_panels.py"),
    ("glm_tpu/optimized/prefill_routes.py", "glm_tpu/greenfield/kernels/prefill_routes.py"),
    ("glm_tpu/optimized/ws32_io.py", "glm_tpu/greenfield/kernels/ws32_io.py"),
    ("glm_tpu/optimized/request_session.py", "glm_tpu/greenfield/runtime/ws32_request_session.py"),
    ("glm_tpu/optimized/reference/attention.py", "glm_tpu/greenfield/kernels/reference/attention.py"),
    ("glm_tpu/optimized/reference/dsa.py", "glm_tpu/greenfield/kernels/reference/dsa.py"),
    ("glm_tpu/optimized/reference/fp8.py", "glm_tpu/greenfield/kernels/reference/fp8.py"),
    ("glm_tpu/optimized/reference/linear.py", "glm_tpu/greenfield/kernels/reference/linear.py"),
    ("glm_tpu/optimized/reference/moe.py", "glm_tpu/greenfield/kernels/reference/moe.py"),
    ("glm_tpu/optimized/reference/prefill_index.py", "glm_tpu/greenfield/kernels/reference/prefill_index.py"),
    ("glm_tpu/optimized/reference/rmsnorm.py", "glm_tpu/greenfield/kernels/reference/rmsnorm.py"),
    ("glm_tpu/optimized/reference/rotary.py", "glm_tpu/greenfield/kernels/reference/rotary.py"),
    ("glm_tpu/optimized/reference/dsa_host_rope.py", "glm_tpu/greenfield/kernels/reference/dsa_host_rope.py"),
)
# ... and the definitions split out of research modules (the remainder is archived).
S2F_SPLIT = (
    ("glm_tpu/optimized/sparse_attention.py", "glm_tpu/greenfield/kernels/pallas/sparse_attention.py",
     ("_FINITE_MASK_VALUE", "_TPU_V4_DMA_ROWS", "SparseMlaConfig", "pregathered_sparse_mla_pallas",)),
    ("glm_tpu/optimized/ws32.py", "glm_tpu/greenfield/kernels/ws32.py",
     ("ws32_rms_norm_mapped", "ws32_fused_add_rms_norm_mapped", "ws32_router_from_shards_mapped",)),
    ("glm_tpu/optimized/ws32_layer.py", "glm_tpu/greenfield/kernels/ws32_layer.py",
     ("Ws32QkvAWeights", "Ws32DsaWeights", "Ws32AttentionWeights", "Ws32DenseWeights", "Ws32StrategyNdDenseWeights", "Ws32MoeWeights", "Ws32PreparedAttention", "Ws32DsaResult", "Ws32AttentionResult", "Ws32AttentionLayerResult", "Ws32TransformerLayerResult", "Ws32MlpResult",)),
    ("glm_tpu/optimized/ws32_decoder.py", "glm_tpu/greenfield/runtime/ws32_decoder.py",
     ("Ws32LayerWeights", "Ws32DecoderWeights", "Ws32DecoderState", "Ws32DecodeStepResult", "Ws32DecoderConfig", "_qkv_specs", "_attention_specs", "_dsa_specs", "_dense_specs", "_moe_specs", "ws32_decoder_weight_specs", "_fp8_names", "ws32_decoder_weight_names", "_weight_name_leaves", "_bind_weight_name_tree", "bind_ws32_decoder_weights", "ws32_decoder_state_specs", "ws32_decode_result_specs", "_validate_local_state", "WS32_MAIN_ROPE_THETA", "build_ws32_main_rope_table",)),
    ("glm_tpu/optimized/ws32_batched_prefill.py", "glm_tpu/greenfield/runtime/ws32_batched_prefill.py",
     ("Ws32BatchedPrefillState", "Ws32BatchedPrefillResult", "ws32_batched_prefill_state_specs", "_require_config", "ws32_prefill_embedding_mapped", "_all_owners_healthy", "finish_ws32_batched_prefill",)),
    ("glm_tpu/optimized/reference/dsa_association.py", "glm_tpu/greenfield/kernels/reference/dsa_association.py",
     ("KeyNormMode", "affine_key_layer_norm",)),
    ("glm_tpu/optimized/request_session.py", "glm_tpu/greenfield/kernels/ws32_sampling.py",
     ("request_uniform",)),
    ("glm_tpu/optimized/prefill_attention.py", "glm_tpu/greenfield/kernels/ws32_prefill_attention.py",
     ("_require_block",)),
    ("glm_tpu/optimized/prefill_linear.py", "glm_tpu/greenfield/kernels/ws32_prefill_linear.py",
     ("_require_rows",)),
    ("glm_tpu/optimized/prefill_dsa.py", "glm_tpu/greenfield/kernels/ws32_prefill_dsa.py",
     ("PrefillDsaInputs", "Ws32PrefillDsaResult",)),
    ("glm_tpu/optimized/prefill_layer.py", "glm_tpu/greenfield/kernels/ws32_prefill_layer.py",
     ("Ws32PrefillLayerResult", "Ws32PrefillPrefixResult", "ws32_prefill_router_mapped",)),
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
    # Function-local imports name the homes of their day (S2f moved some of them); compared by the
    # names they bind.
    current, baseline = _definition(_current(home), name), _definition(_baseline(origin), name)
    assert _without_import_paths(current) == _without_import_paths(baseline)


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


def test_worker_and_runtime_modules_import_nothing_from_the_research_layer():
    code = """
import json, sys
for name in ("scripts.release.ws32_optimized_worker", "glm_tpu.distributed.parallel_state",
             "glm_tpu.optimized.runtime", "glm_tpu.optimized.batched_runtime", "glm_tpu.optimized.topology_binding",
             "glm_tpu.runner.compilation_manager", "glm_tpu.runner.kv_cache_manager",
             "glm_tpu.optimized.source_inventory"):
    __import__(name)
print(json.dumps(sorted(sys.modules)))
"""
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=REPO,
                                                env=dict(os.environ, JAX_PLATFORMS="cpu"), text=True, timeout=120))
    research = [m for m in loaded if m.startswith("scripts.greenfield")
                or m.startswith(("glm_tpu.greenfield.benchmarking", "glm_tpu.greenfield.validation"))]
    assert research == []


def test_production_sources_name_no_research_home():
    paths = [*(path for path in sorted((REPO / "glm_tpu").rglob("*.py"))
               if "greenfield" not in path.relative_to(REPO).parts),
             REPO / "scripts/release/ws32_optimized_worker.py",
             REPO / "scripts/release/launch_ws32_optimized_request.py",
             REPO / "scripts/release/ws32_pack_worker.py"]
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
                assert not name.startswith(("scripts.greenfield", "glm_tpu.greenfield")), (path.name, name)


def _s2f_base(path: str) -> ast.Module:
    return ast.parse(subprocess.check_output(["git", "show", f"{S2F_BASE}:{path}"], cwd=REPO, text=True))


def _top_level_names(tree: ast.Module) -> list[str]:
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return names


def _without_import_paths(node: ast.AST) -> str:
    """AST dump with every import reduced to the names it binds (a moved module imports the same
    objects from their new homes)."""
    import copy

    node = copy.deepcopy(node)
    for inner in ast.walk(node):
        if isinstance(inner, ast.ImportFrom):
            inner.module, inner.level = "<moved>", 0
    return ast.dump(node)


@pytest.mark.parametrize(("home", "origin"), S2F_WHOLE)
def test_s2f_moved_file_keeps_every_definition(home, origin):
    base, current = _s2f_base(origin), _current(home)
    names = _top_level_names(base)
    # same definitions in the same order (a split may append its definitions to a moved file)
    assert [n for n in _top_level_names(current) if n in names] == names
    for name in names:
        assert _without_import_paths(_definition(current, name)) == _without_import_paths(_definition(base, name))


@pytest.mark.parametrize(("home", "origin", "names"), S2F_SPLIT)
def test_s2f_split_definitions_equal_their_research_originals(home, origin, names):
    base, current = _s2f_base(origin), _current(home)
    for name in names:
        assert _without_import_paths(_definition(current, name)) == _without_import_paths(_definition(base, name))


def test_production_imports_nothing_from_the_research_package():
    code = """
import json, sys
for name in ("glm_tpu.cli", "glm_tpu.api", "glm_tpu.ui", "glm_tpu.optimized.ask", "glm_tpu.optimized.runtime",
             "glm_tpu.optimized.batched_runtime", "glm_tpu.optimized.checkpoint", "glm_tpu.optimized.admission",
             "glm_tpu.runner.programs", "scripts.release.launch_ws32_optimized_request",
             "scripts.release.ws32_optimized_worker", "scripts.release.ws32_pack_worker"):
    __import__(name)
print(json.dumps(sorted(sys.modules)))
"""
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=REPO,
                                                env=dict(os.environ, JAX_PLATFORMS="cpu"), text=True, timeout=120))
    assert [m for m in loaded if m.startswith(("glm_tpu.greenfield", "scripts.greenfield"))] == []
