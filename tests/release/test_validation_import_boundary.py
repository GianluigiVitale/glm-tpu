"""Validate lazy package exports without changing any validator implementation."""

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
PATH = "glm_tpu/greenfield/validation/__init__.py"
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"


def isolated_package():
    spec = importlib.util.spec_from_file_location(
        "validation_fixture", REPO / PATH, submodule_search_locations=[]
    )
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    return package


def test_original_named_exports_and_order_except_unbound_alias():
    source = subprocess.check_output(
        ["git", "show", BASE + ":" + PATH], cwd=REPO, text=True
    )
    tree = ast.parse(source)
    expected = {
        alias.asname or alias.name: node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    original_all = ast.literal_eval(
        next(n.value for n in tree.body if isinstance(n, ast.Assign))
    )
    assert set(original_all) - set(expected) == {"Ws32LongContextOracle"}
    package = isolated_package()
    assert package._EXPORTS == expected
    assert package.__all__ == tuple(n for n in original_all if n in expected)
    for module in set(expected.values()):
        assert (REPO / PATH).with_name(module + ".py").is_file()


def test_lazy_dispatch_preserves_identity_caches_and_refuses_unknown(monkeypatch):
    package = isolated_package()
    result, calls = object(), []

    def importer(module, parent):
        calls.append((module, parent))
        return SimpleNamespace(validate_ws32_cache_probe=result)

    monkeypatch.setattr(package, "_import_module", importer)
    assert package.validate_ws32_cache_probe is result
    assert package.validate_ws32_cache_probe is result
    assert calls == [(".ws32_short_context", "validation_fixture")]
    assert "validate_ws32_cache_probe" in dir(package)
    for name in ("missing_export", "Ws32LongContextOracle"):
        with pytest.raises(AttributeError):
            getattr(package, name)
    assert len(calls) == 1


def test_bare_import_is_stdlib_only():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import glm_tpu.greenfield.validation; "
        "assert not any(n.startswith('glm_tpu.greenfield.validation.') for n in sys.modules); "
        "assert not any(n.split('.')[0] in {'jax', 'jaxlib', 'torch'} for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(REPO)],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=10,
    )


def test_actual_named_and_submodule_imports_do_not_load_unrelated_oracles():
    code = """
import sys
from glm_tpu.greenfield.validation import validate_ws32_cache_probe, ws32_prefill_memory
from glm_tpu.greenfield.validation.ws32_short_context import validate_ws32_cache_probe as direct
assert validate_ws32_cache_probe is direct
assert ws32_prefill_memory.__name__.endswith('.ws32_prefill_memory')
assert 'glm_tpu.greenfield.validation.legacy_residuals' not in sys.modules
assert 'glm_tpu.greenfield.validation.legacy_main_cache' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        timeout=20,
    )


def test_every_retained_export_resolves_to_its_real_module_attribute():
    code = """
import importlib, json
import glm_tpu.greenfield.validation as package
for name, module in package._EXPORTS.items():
    direct = importlib.import_module('.' + module, package.__name__)
    assert getattr(package, name) is getattr(direct, name), name
namespace = {}
exec('from glm_tpu.greenfield.validation import *', namespace)
assert set(namespace) - {'__builtins__'} == set(package.__all__)
print(json.dumps(len(package._EXPORTS)))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        text=True,
        timeout=20,
    )
    assert json.loads(output) == len(isolated_package()._EXPORTS)
