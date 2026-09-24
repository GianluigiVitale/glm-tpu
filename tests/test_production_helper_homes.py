"""S2a/S2f: the production helpers live in production modules, moved verbatim.

Every moved definition equals its original by AST (the S2a helpers their 181c013e definitions, the
S2f files and splits their definitions at the S2f base); the two file writers lose only their
research-namespace branches, which never match the worker's HLO directory. The origins are read
from those commits by their paths there (the research tree is at ``archive/research-20260922``);
the homes below are the S3 paths (``tools/migration/move_map.toml``), every definition S4.1
moved on is found at its final home (``tools/migration/symbol_moves.toml``, which also names the
collision renames), and every definition S4.2 renamed under its public name
(``tools/migration/renames.toml``; references to renamed definitions are compared as renamed).
Production loads and imports only ``glm_tpu`` modules of this tree.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
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
    ("glm_tpu/distributed/topology.py", RUN_SHORT, "_device_record"),
    ("glm_tpu/distributed/topology.py", ONE_LAYER, "_TOPOLOGY_CAPTURE_KEYS"),
    ("glm_tpu/distributed/topology.py", ONE_LAYER, "validate_ws32_topology_fleet"),
    ("glm_tpu/runner/compilation_manager.py", ORIGINALS, "compile_program"),
    ("glm_tpu/runner/compilation_manager.py", MICROBENCH, "_compiled_memory"),
    ("glm_tpu/runner/compilation_manager.py", MICROBENCH, "_memory_stats"),
    ("glm_tpu/runner/kv_cache_manager.py", NATIVE_PROGRAMS, "build_cache_initializer"),
    ("glm_tpu/models/glm_moe_dsa/weights.py", ORIGINALS, "build_wk_programs"),
    ("glm_tpu/model_loader/source_inventory.py", ORIGINALS, "authenticated_inventory"),
    ("glm_tpu/layers/attention/kv_cache.py", STAGE_LOCAL, "_require_decode_metadata"),
)


# S2f: the production definitions of the research package moved into production modules, verbatim.
# Whole files (every top-level definition of the S2f base file) ...
S2F_BASE = "c8ee4fff5003d74cd8a75b68135a8f4bf7951280"
S2F_WHOLE = (
    ("glm_tpu/exceptions.py", "glm_tpu/greenfield/errors.py"),
    ("glm_tpu/config/model.py", "glm_tpu/greenfield/types.py"),
    ("glm_tpu/distributed/mesh.py", "glm_tpu/greenfield/sharding/ws32.py"),
    ("glm_tpu/runner/hlo_utils.py", "glm_tpu/greenfield/sharding/hlo_contract.py"),
    ("glm_tpu/model_loader/source_inventory.py", "glm_tpu/greenfield/partitioning/source_inventory.py"),
    ("glm_tpu/model_loader/placement.py", "glm_tpu/greenfield/checkpoint/ws32_runtime.py"),
    ("glm_tpu/model_loader/sharded_state/format.py", "glm_tpu/greenfield/checkpoint/ws32_runtime_checkpoint.py"),
    ("glm_tpu/kernels/fp8_grouped_matmul/panel_kernel.py", "glm_tpu/greenfield/kernels/pallas/prefill_panel_fp8.py"),
    ("glm_tpu/layers/attention/kv_cache.py", "glm_tpu/greenfield/kernels/prefill_cache.py"),
    ("glm_tpu/kernels/fp8_grouped_matmul/panels.py", "glm_tpu/greenfield/kernels/prefill_expert_panels.py"),
    ("glm_tpu/layers/moe/_s3_prefill_routes.py", "glm_tpu/greenfield/kernels/prefill_routes.py"),
    ("glm_tpu/layers/sampler.py", "glm_tpu/greenfield/kernels/ws32_io.py"),
    ("glm_tpu/engine/request_session.py", "glm_tpu/greenfield/runtime/ws32_request_session.py"),
    ("glm_tpu/layers/attention/_s3_attention.py", "glm_tpu/greenfield/kernels/reference/attention.py"),
    ("glm_tpu/layers/attention/_s3_dsa.py", "glm_tpu/greenfield/kernels/reference/dsa.py"),
    ("glm_tpu/layers/fp8.py", "glm_tpu/greenfield/kernels/reference/fp8.py"),
    ("glm_tpu/layers/_s3_linear.py", "glm_tpu/greenfield/kernels/reference/linear.py"),
    ("glm_tpu/layers/moe/router.py", "glm_tpu/greenfield/kernels/reference/moe.py"),
    ("glm_tpu/layers/_s3_prefill_index.py", "glm_tpu/greenfield/kernels/reference/prefill_index.py"),
    ("glm_tpu/layers/_s3_rmsnorm.py", "glm_tpu/greenfield/kernels/reference/rmsnorm.py"),
    ("glm_tpu/layers/rope.py", "glm_tpu/greenfield/kernels/reference/rotary.py"),
    ("glm_tpu/layers/_s3_dsa_host_rope.py", "glm_tpu/greenfield/kernels/reference/dsa_host_rope.py"),
)
# ... and the definitions split out of research modules (the remainder is archived).
S2F_SPLIT = (
    ("glm_tpu/kernels/sparse_mla/kernel.py", "glm_tpu/greenfield/kernels/pallas/sparse_attention.py",
     ("_FINITE_MASK_VALUE", "_TPU_V4_DMA_ROWS", "SparseMlaConfig", "pregathered_sparse_mla_pallas",)),
    ("glm_tpu/layers/norm.py", "glm_tpu/greenfield/kernels/ws32.py",
     ("ws32_rms_norm_mapped", "ws32_fused_add_rms_norm_mapped", "ws32_router_from_shards_mapped",)),
    ("glm_tpu/models/glm_moe_dsa/_s3_ws32_layer.py", "glm_tpu/greenfield/kernels/ws32_layer.py",
     ("Ws32QkvAWeights", "Ws32DsaWeights", "Ws32AttentionWeights", "Ws32DenseWeights", "Ws32StrategyNdDenseWeights", "Ws32MoeWeights", "Ws32PreparedAttention", "Ws32DsaResult", "Ws32AttentionResult", "Ws32AttentionLayerResult", "Ws32TransformerLayerResult", "Ws32MlpResult",)),
    ("glm_tpu/models/glm_moe_dsa/_s3_ws32_decoder.py", "glm_tpu/greenfield/runtime/ws32_decoder.py",
     ("Ws32LayerWeights", "Ws32DecoderWeights", "Ws32DecoderState", "Ws32DecodeStepResult", "Ws32DecoderConfig", "_qkv_specs", "_attention_specs", "_dsa_specs", "_dense_specs", "_moe_specs", "ws32_decoder_weight_specs", "_fp8_names", "ws32_decoder_weight_names", "_weight_name_leaves", "_bind_weight_name_tree", "bind_ws32_decoder_weights", "ws32_decoder_state_specs", "ws32_decode_result_specs", "_validate_local_state", "WS32_MAIN_ROPE_THETA", "build_ws32_main_rope_table",)),
    ("glm_tpu/models/glm_moe_dsa/state.py", "glm_tpu/greenfield/runtime/ws32_batched_prefill.py",
     ("Ws32BatchedPrefillState", "Ws32BatchedPrefillResult", "ws32_batched_prefill_state_specs", "_require_config", "ws32_prefill_embedding_mapped", "_all_owners_healthy", "finish_ws32_batched_prefill",)),
    ("glm_tpu/layers/_s3_dsa_association.py", "glm_tpu/greenfield/kernels/reference/dsa_association.py",
     ("KeyNormMode", "affine_key_layer_norm",)),
    ("glm_tpu/engine/request_session.py", "glm_tpu/greenfield/kernels/ws32_sampling.py",
     ("request_uniform",)),
    ("glm_tpu/layers/attention/mla.py", "glm_tpu/greenfield/kernels/ws32_prefill_attention.py",
     ("_require_block",)),
    ("glm_tpu/layers/_s3_prefill_linear.py", "glm_tpu/greenfield/kernels/ws32_prefill_linear.py",
     ("_require_rows",)),
    ("glm_tpu/layers/attention/_s3_prefill_dsa.py", "glm_tpu/greenfield/kernels/ws32_prefill_dsa.py",
     ("PrefillDsaInputs", "Ws32PrefillDsaResult",)),
    ("glm_tpu/models/glm_moe_dsa/decoder_layer.py", "glm_tpu/greenfield/kernels/ws32_prefill_layer.py",
     ("Ws32PrefillLayerResult", "Ws32PrefillPrefixResult", "ws32_prefill_router_mapped",)),
)


def _symbol_moves() -> dict:
    import tomllib

    return tomllib.loads((REPO / "tools/migration/symbol_moves.toml").read_text())


def _public_names() -> dict:
    """S4.2: ``final home:name`` -> public name (``tools/migration/renames.toml``)."""
    import tomllib

    return tomllib.loads((REPO / "tools/migration/renames.toml").read_text())["renames"]


def _final_home(home: str, name: str) -> tuple[str, str]:
    """Where a definition of the S3 file ``home`` is now, under which name."""
    table = _symbol_moves()
    spec = table["moves"].get(home, {}).get(name)
    if spec is None:
        path, current = home, table.get("renames", {}).get(f"{home}:{name}", name)
    else:
        path, current = (spec, name) if isinstance(spec, str) else (spec["to"], spec.get("as", name))
    return path, _public_names().get(f"{path}:{current}", current)


def _as_renamed_in(home: str, node: ast.AST) -> ast.AST:
    """``node`` (an original definition of ``home``) with its references to the definitions S4.1
    renamed in ``home`` and to every definition S4.2 renamed spelled as renamed."""
    import copy

    renames = {key.partition(":")[2]: new for key, new in _symbol_moves().get("renames", {}).items()
               if key.partition(":")[0] == home}
    public = {key.partition(":")[2]: new for key, new in _public_names().items()}
    node = copy.deepcopy(node)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name):
            name = renames.get(inner.id, inner.id)
            inner.id = public.get(name, name)
        elif isinstance(inner, ast.ImportFrom):  # the name an import binds (an ``as`` alias stays)
            for alias in inner.names:
                alias.name = public.get(alias.name, alias.name)
    return node


def _moved_definition(home: str, name: str) -> ast.AST:
    """The definition at its final home, under its original name (a moved definition is verbatim
    up to the name a collision gave it)."""
    import copy

    path, current_name = _final_home(home, name)
    node = copy.deepcopy(_definition(_current(path), current_name))
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        node.name = name
    return node


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
    current, baseline = _moved_definition(home, name), _as_renamed_in(home, _definition(_baseline(origin), name))
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


def _repository_modules(names: tuple[str, ...]) -> list[str]:
    """Every module loaded from this tree after importing ``names`` in a fresh interpreter."""
    code = f"""
import json, sys
for name in {names!r}:
    __import__(name)
root = {str(REPO) + "/"!r}
print(json.dumps(sorted(n for n, m in list(sys.modules.items())
                        if str(getattr(m, "__file__", None) or "").startswith(root))))
"""
    return json.loads(subprocess.check_output([sys.executable, "-c", code], cwd=REPO,
                                              env=dict(os.environ, JAX_PLATFORMS="cpu"), text=True, timeout=120))


def test_worker_and_runtime_modules_load_only_glm_tpu_modules():
    loaded = _repository_modules(("glm_tpu.worker.tpu_worker", "glm_tpu.distributed.parallel_state",
                                  "glm_tpu.runner.tpu_runner", "glm_tpu.engine.llm_engine",
                                  "glm_tpu.distributed.topology", "glm_tpu.runner.compilation_manager",
                                  "glm_tpu.runner.kv_cache_manager", "glm_tpu.model_loader.source_inventory"))
    assert loaded and [m for m in loaded if m != "glm_tpu" and not m.startswith("glm_tpu.")] == []


def _is_module(name: str) -> bool:
    path = REPO / name.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_production_sources_import_only_existing_glm_tpu_modules():
    for path in sorted((REPO / "glm_tpu").rglob("*.py")):
        package = path.relative_to(REPO).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = list(package[:len(package) - node.level + 1]) if node.level else []
                names = [".".join([*base, *([node.module] if node.module else [])])]
            for name in names:
                top = name.split(".", 1)[0]
                assert top not in ("scripts", "tools", "tests", "bench", "benchmarks", "examples"), (path.name, name)
                assert top != "glm_tpu" or _is_module(name), (path.name, name)


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


def _pruned_at_s2f(home: str) -> set[str]:
    """Dead definitions S2f step 5 pruned from ``home`` (the reviewed plan)."""
    import tomllib

    plan = tomllib.loads((REPO / "tools/migration/keep_symbols.toml").read_text())
    moves = tomllib.loads((REPO / "tools/migration/move_map.toml").read_text())["files"]
    home = next((old for old, new in moves.items() if new == home), home)  # its S2f path (before S3)
    module = home[:-3].replace("/", ".")
    return set(plan.get(module, {}).get("prune", []))


@pytest.mark.parametrize(("home", "origin"), S2F_WHOLE)
def test_s2f_moved_file_keeps_every_definition(home, origin):
    base = _s2f_base(origin)
    names = [n for n in _top_level_names(base) if n not in _pruned_at_s2f(home)]
    # the definitions that stay in the file keep their order, minus the pruned dead ones (a split
    # may append its definitions to a moved file); S4.1 moved the others on, verbatim
    staying = [n for n in names if _final_home(home, n)[0] == home]
    if staying:
        current = [_final_home(home, n)[1] for n in staying]
        assert [n for n in _top_level_names(_current(home)) if n in current] == current
    for name in names:
        original = _as_renamed_in(home, _definition(base, name))
        assert _without_import_paths(_moved_definition(home, name)) == _without_import_paths(original)


@pytest.mark.parametrize(("home", "origin", "names"), S2F_SPLIT)
def test_s2f_split_definitions_equal_their_research_originals(home, origin, names):
    base = _s2f_base(origin)
    for name in names:
        original = _as_renamed_in(home, _definition(base, name))
        assert _without_import_paths(_moved_definition(home, name)) == _without_import_paths(original)


def test_every_entry_point_loads_only_glm_tpu_modules():
    loaded = _repository_modules((
        "glm_tpu.entrypoints.cli.main", "glm_tpu.entrypoints.openai.serving_chat", "glm_tpu.entrypoints.serve.server",
        "glm_tpu.entrypoints.cli.ask", "glm_tpu.runner.tpu_runner", "glm_tpu.engine.llm_engine",
        "glm_tpu.model_loader.sharded_state.manifest", "glm_tpu.runner.admission", "glm_tpu.runner.programs",
        "glm_tpu.executor.multihost_executor", "glm_tpu.worker.tpu_worker", "glm_tpu.model_loader.pack_worker"))
    assert loaded and [m for m in loaded if m != "glm_tpu" and not m.startswith("glm_tpu.")] == []
