"""The native engine must not import legacy execution (vLLM, tpu-inference), and its packages keep the layering
of DESIGN 8.7: ``layers`` never imports ``models``, ``engine`` never imports ``entrypoints``, and ``config`` imports
only the leaf modules below it and no JAX (S5 WU-S3 moved the definitions that crossed these lines down).
Production loads and imports only ``glm_tpu`` modules of this tree: no ``tools``, ``tests``, ``scripts`` or other
repository code, and no ``glm_tpu`` module that does not exist.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "glm_tpu"


def test_engine_has_no_legacy_or_vllm_imports() -> None:
    forbidden = {"tpu_inference", "vllm"}
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            if roots & forbidden:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


def _imported_modules(path: Path) -> set[str]:
    """Every module an import statement of ``path`` names, at any depth (``from m import a`` names ``m`` and
    ``m.a``, which may be a submodule)."""
    package = path.relative_to(ROOT).with_suffix("").parts[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[: len(package) - node.level + 1]) if node.level else ""
            module = ".".join(part for part in (base, node.module or "") if part)
            names |= {module} | {f"{module}.{alias.name}" for alias in node.names}
    return names


def _within(name: str, root: str) -> bool:
    return name == root or name.startswith(root + ".")


def _offenders(package: str, allowed) -> list[str]:
    """``path: module`` for each ``glm_tpu`` module a file of ``glm_tpu/<package>`` imports and ``allowed`` refuses."""
    return [
        f"{path.relative_to(ROOT)}: {name}"
        for path in sorted((PACKAGE / package).rglob("*.py"))
        for name in sorted(_imported_modules(path))
        if name.startswith("glm_tpu.") and not allowed(name)
    ]


def test_layers_never_import_models() -> None:
    # The BF16 weight groups the layer bodies take are defined in glm_tpu.layers.contracts.
    assert _offenders("layers", lambda name: not _within(name, "glm_tpu.models")) == []


def test_engine_never_imports_entrypoints() -> None:
    # The resident client raises glm_tpu.exceptions.ApiError and splits answers with engine.outputs.final_channel.
    assert _offenders("engine", lambda name: not _within(name, "glm_tpu.entrypoints")) == []


def test_config_imports_only_leaf_modules_and_no_jax() -> None:
    # config sits below distributed, kernels, layers and models; it defines the contracts the layers re-export.
    leaves = ("glm_tpu.config", "glm_tpu.exceptions", "glm_tpu.utils", "glm_tpu.envs")
    assert _offenders("config", lambda name: any(_within(name, root) for root in leaves)) == []
    modules = sorted(
        ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        for path in (PACKAGE / "config").rglob("*.py")
    )
    code = "import sys\n" + "".join(f"import {name}\n" for name in modules)
    code += "print(sorted(name for name in ('jax', 'jaxlib') if name in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    # Five on purpose (the package, cache, model, parallel, site): a new config module fails this count, the prompt
    # to extend it (the leaf and no-JAX checks above then cover the new module too); it is not a regression.
    assert len(modules) == 5 and result.stdout == "[]\n", (modules, result.stdout)


def test_moved_definitions_are_reexported_as_the_same_objects() -> None:
    # Their former homes, which the rest of the tree imports them from, bind the definitions themselves.
    from glm_tpu import exceptions
    from glm_tpu.config import cache
    from glm_tpu.engine import outputs
    from glm_tpu.entrypoints.openai import protocol, tool_parser
    from glm_tpu.layers import contracts
    from glm_tpu.models.glm_moe_dsa import weights

    assert protocol.ApiError is exceptions.ApiError and exceptions.ApiError.__module__ == "glm_tpu.exceptions"
    assert tool_parser.final_channel is outputs.final_channel and outputs.final_channel.__module__ == outputs.__name__
    for name in ("MlaNumericalContract", "StageLocalKvLayout", "DsaNumericalContract", "GlmMoeNumericalContract"):
        assert getattr(contracts, name) is getattr(cache, name) and getattr(cache, name).__module__ == cache.__name__
    for name in ("Bf16QkvAWeights", "Bf16AttentionWeights", "Bf16DsaWeights", "Bf16DenseWeights"):
        defined = getattr(contracts, name)
        assert getattr(weights, name) is defined and defined.__module__ == contracts.__name__


def _repository_modules(names: tuple[str, ...]) -> list[str]:
    """Every module loaded from this tree after importing ``names`` in a fresh interpreter."""
    code = f"""
import json, sys
for name in {names!r}:
    __import__(name)
root = {str(ROOT) + "/"!r}
print(json.dumps(sorted(n for n, m in list(sys.modules.items())
                        if str(getattr(m, "__file__", None) or "").startswith(root))))
"""
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code], cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu"), text=True, timeout=120
        )
    )


def test_worker_and_runtime_modules_load_only_glm_tpu_modules():
    loaded = _repository_modules(
        (
            "glm_tpu.worker.tpu_worker",
            "glm_tpu.distributed.parallel_state",
            "glm_tpu.runner.tpu_runner",
            "glm_tpu.engine.llm_engine",
            "glm_tpu.distributed.topology",
            "glm_tpu.runner.compilation_manager",
            "glm_tpu.runner.kv_cache_manager",
            "glm_tpu.model_loader.source_inventory",
        )
    )
    assert loaded and [m for m in loaded if m != "glm_tpu" and not m.startswith("glm_tpu.")] == []


def _is_module(name: str) -> bool:
    path = ROOT / name.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_production_sources_import_only_existing_glm_tpu_modules():
    for path in sorted((ROOT / "glm_tpu").rglob("*.py")):
        package = path.relative_to(ROOT).with_suffix("").parts[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = list(package[: len(package) - node.level + 1]) if node.level else []
                names = [".".join([*base, *([node.module] if node.module else [])])]
            for name in names:
                top = name.split(".", 1)[0]
                assert top not in ("scripts", "tools", "tests", "bench", "benchmarks", "examples"), (path.name, name)
                assert top != "glm_tpu" or _is_module(name), (path.name, name)


def test_every_entry_point_loads_only_glm_tpu_modules():
    loaded = _repository_modules(
        (
            "glm_tpu.entrypoints.cli.main",
            "glm_tpu.entrypoints.openai.serving_chat",
            "glm_tpu.entrypoints.serve.server",
            "glm_tpu.entrypoints.cli.ask",
            "glm_tpu.runner.tpu_runner",
            "glm_tpu.engine.llm_engine",
            "glm_tpu.model_loader.sharded_state.manifest",
            "glm_tpu.runner.admission",
            "glm_tpu.runner.programs",
            "glm_tpu.executor.multihost_executor",
            "glm_tpu.worker.tpu_worker",
            "glm_tpu.model_loader.pack_worker",
        )
    )
    assert loaded and [m for m in loaded if m != "glm_tpu" and not m.startswith("glm_tpu.")] == []
