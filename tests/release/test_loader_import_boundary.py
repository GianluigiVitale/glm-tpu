"""Lazy checkpoint/partitioning facades must not load PP-era packs for WS32 loads.

Both packages keep every originally bound name and the exact ordered __all__
from starting main; only the moment of import changes. No loader body,
checkpoint identity or partition contract changes here.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"
PACKAGES = {
    "checkpoint": "glm_tpu/greenfield/checkpoint/__init__.py",
    "partitioning": "glm_tpu/greenfield/partitioning/__init__.py",
}
RETIRED_MODULES = {
    "checkpoint": set(["full_loader", "gate_c", "gate_c_loader", "one_layer", "one_layer_loader", "one_layer_pallas", "one_layer_pallas_feature", "one_layer_pallas_feature_loader", "one_layer_pallas_loader", "runtime_feature", "runtime_feature_loader", "runtime_loader", "runtime_pack", "stream_pack", "ws32_one_layer"]),
    "partitioning": set(["layer_assigner", "manifest", "memory_model", "ownership"]),
}
PP_ERA_CHILDREN = (
    "glm_tpu.greenfield.checkpoint.full_loader",
    "glm_tpu.greenfield.checkpoint.gate_c",
    "glm_tpu.greenfield.checkpoint.gate_c_loader",
    "glm_tpu.greenfield.checkpoint.one_layer",
    "glm_tpu.greenfield.checkpoint.one_layer_loader",
    "glm_tpu.greenfield.checkpoint.runtime_loader",
    "glm_tpu.greenfield.checkpoint.runtime_pack",
    "glm_tpu.greenfield.checkpoint.stream_pack",
    "glm_tpu.greenfield.partitioning.layer_assigner",
    "glm_tpu.greenfield.partitioning.manifest",
    "glm_tpu.greenfield.partitioning.memory_model",
    "glm_tpu.greenfield.partitioning.ownership",
    "glm_tpu.greenfield.topology",
)


def isolated_package(name):
    spec = importlib.util.spec_from_file_location(
        name + "_fixture", REPO / PACKAGES[name], submodule_search_locations=[]
    )
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    return package


def baseline_bindings(name):
    source = subprocess.check_output(
        ["git", "show", BASE + ":" + PACKAGES[name]], cwd=REPO, text=True
    )
    tree = ast.parse(source)
    expected = {}
    exports = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 1
            for alias in node.names:
                bound = alias.asname or alias.name
                target = node.module
                if alias.name != bound:
                    target += "." + alias.name
                expected[bound] = target
        elif isinstance(node, ast.Assign):
            assert node.targets[0].id == "__all__"
            exports = ast.literal_eval(node.value)
    return expected, exports


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_retained_named_exports_and_ordered_all_match_baseline(name):
    expected, exports = baseline_bindings(name)
    retired = RETIRED_MODULES[name]
    retained = {
        bound: target
        for bound, target in expected.items()
        if target.partition(".")[0] not in retired
    }
    package = isolated_package(name)
    assert package._EXPORTS == retained
    assert list(package.__all__) == [n for n in exports if n in retained]
    assert type(package.__all__) is type(exports)
    assert set(exports) == set(expected)
    for target in set(retained.values()):
        module = target.partition(".")[0]
        assert (REPO / PACKAGES[name]).with_name(module + ".py").is_file()
    for module in retired:
        assert not (REPO / PACKAGES[name]).with_name(module + ".py").exists()


def test_partitioning_alias_left_with_the_retired_manifest_module():
    expected, _ = baseline_bindings("partitioning")
    assert expected["LAYOUT_ARTIFACT_KIND"] == "manifest.ARTIFACT_KIND"
    package = isolated_package("partitioning")
    assert "LAYOUT_ARTIFACT_KIND" not in package._EXPORTS
    with pytest.raises(AttributeError):
        package.LAYOUT_ARTIFACT_KIND


@pytest.mark.parametrize(
    ("name", "attribute", "module", "target"),
    (
        ("checkpoint", "verify_ws32_runtime_checkpoint", ".ws32_runtime_checkpoint", "verify_ws32_runtime_checkpoint"),
        ("partitioning", "SourceInventory", ".source_inventory", "SourceInventory"),
    ),
)
def test_lazy_dispatch_preserves_identity_caches_and_refuses_unknown(
    monkeypatch, name, attribute, module, target
):
    package = isolated_package(name)
    result, calls = object(), []

    def importer(child, parent):
        calls.append((child, parent))
        return SimpleNamespace(**{target: result})

    monkeypatch.setattr(package, "_import_module", importer)
    assert getattr(package, attribute) is result
    assert getattr(package, attribute) is result
    assert calls == [(module, name + "_fixture")]
    assert attribute in dir(package)
    with pytest.raises(AttributeError):
        package.no_such_export
    assert len(calls) == 1


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_bare_import_is_stdlib_only(name):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        f"import glm_tpu.greenfield.{name}; "
        f"assert not any(n.startswith('glm_tpu.greenfield.{name}.') for n in sys.modules); "
        "assert not any(n.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch'} for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(REPO)],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=10,
    )


def test_ws32_loader_imports_do_not_load_pp_era_packs_or_topology():
    code = """
import json, sys
from glm_tpu.greenfield.checkpoint import (
    load_ws32_strategy_nd_dense_overlay,
    verify_ws32_runtime_checkpoint,
)
from glm_tpu.greenfield.partitioning import SourceInventory
from glm_tpu.greenfield.partitioning.source_inventory import SourceInventory as direct_inventory
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
    verify_ws32_runtime_checkpoint as direct,
)
assert verify_ws32_runtime_checkpoint is direct
assert SourceInventory is direct_inventory
print(json.dumps(sorted(n for n in sys.modules if n.startswith('glm_tpu.greenfield.'))))
"""
    loaded = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=REPO,
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            text=True,
            timeout=30,
        )
    )
    assert "glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint" in loaded
    assert "glm_tpu.greenfield.partitioning.source_inventory" in loaded
    assert not set(loaded) & set(PP_ERA_CHILDREN)
    assert not any(n.startswith("glm_tpu.greenfield.topology") for n in loaded)


def test_ws32_checkpoint_only_import_loads_no_partitioning_beyond_source_inventory():
    code = """
import json, sys
from glm_tpu.greenfield.checkpoint import verify_ws32_runtime_checkpoint
print(json.dumps(sorted(n for n in sys.modules if n.startswith('glm_tpu.greenfield.'))))
"""
    loaded = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=REPO,
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            text=True,
            timeout=30,
        )
    )
    assert not set(loaded) & set(PP_ERA_CHILDREN)
    assert not any(n.startswith("glm_tpu.greenfield.topology") for n in loaded)
    partitioning = [n for n in loaded if n.startswith("glm_tpu.greenfield.partitioning.")]
    assert partitioning == ["glm_tpu.greenfield.partitioning.source_inventory"]


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_every_export_resolves_to_its_real_module_attribute(name):
    code = f"""
import importlib, json
import glm_tpu.greenfield.{name} as package
for bound, target in package._EXPORTS.items():
    module, _, attribute = target.partition('.')
    direct = importlib.import_module('.' + module, package.__name__)
    assert getattr(package, bound) is getattr(direct, attribute or bound), bound
namespace = {{}}
exec('from glm_tpu.greenfield.{name} import *', namespace)
assert set(namespace) - {{'__builtins__'}} == set(package.__all__)
print(json.dumps(len(package._EXPORTS)))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        text=True,
        timeout=60,
    )
    assert json.loads(output) == len(isolated_package(name)._EXPORTS)
