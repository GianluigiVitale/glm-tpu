"""The kept greenfield package initializers import nothing (S2b); WS32 loads stay narrow.

Every name is imported from the submodule that defines it, so importing a package runs only its
docstring, and the WS32 checkpoint loader never loads the retired PP-era packs. No loader body,
checkpoint identity or partition contract changes here.
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
# DESIGN 10.3 S2b: the package initializers made import-free.
PACKAGES = ("runtime", "kernels", "kernels.pallas", "kernels.reference", "sharding", "checkpoint", "partitioning")
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


def _initializer(name: str) -> Path:
    return REPO / "glm_tpu/greenfield" / name.replace(".", "/") / "__init__.py"


@pytest.mark.parametrize("name", PACKAGES)
def test_package_initializer_is_a_docstring(name):
    tree = ast.parse(_initializer(name).read_text())
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr), name
    assert isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str), name


@pytest.mark.parametrize("name", sorted(RETIRED_MODULES))
def test_retired_pp_era_modules_stay_retired(name):
    for module in RETIRED_MODULES[name]:
        assert not _initializer(name).with_name(module + ".py").exists(), module


@pytest.mark.parametrize("name", PACKAGES)
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
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import verify_ws32_runtime_checkpoint
from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import load_ws32_strategy_nd_dense_overlay
from glm_tpu.greenfield.partitioning.source_inventory import SourceInventory
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
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import verify_ws32_runtime_checkpoint
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
