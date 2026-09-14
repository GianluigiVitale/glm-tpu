"""Do not initialize every research benchmark when importing one inspector."""

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
PATH = "glm_tpu/greenfield/benchmarking/__init__.py"
BASE = "b667f00f1ae48c8ff37e92500550c1395d74c66d"


def isolated_package():
    spec = importlib.util.spec_from_file_location(
        "benchmark_fixture", REPO / PATH, submodule_search_locations=[]
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_all_original_named_export_targets_preserved():
    source = subprocess.check_output(
        ["git", "show", BASE + ":" + PATH], cwd=REPO, text=True
    )
    expected = {}
    exports = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 1
            expected.update({a.asname or a.name: node.module for a in node.names})
        elif isinstance(node, ast.Assign):
            exports = ast.literal_eval(node.value)
    package = isolated_package()
    assert package._EXPORTS == expected
    assert package.__all__ == exports


def test_exact_attribute_identity_cached_and_unknown_refused(monkeypatch):
    package = isolated_package()
    result = object()
    calls = []

    def importer(module, parent):
        calls.append((module, parent))
        return SimpleNamespace(validate_ws32_decoder_hlo=result)

    monkeypatch.setattr(package, "_import_module", importer)
    assert package.validate_ws32_decoder_hlo is result
    assert package.validate_ws32_decoder_hlo is result
    assert calls == [(".ws32_decoder", "benchmark_fixture")]
    assert "validate_ws32_decoder_hlo" in dir(package)
    with pytest.raises(AttributeError):
        package.no_such_export


def test_bare_package_does_not_load_research_modules_or_jax():
    program = """
import json, sys
import glm_tpu.greenfield.benchmarking
prefix = 'glm_tpu.greenfield.benchmarking.'
print(json.dumps({'children': sorted(n for n in sys.modules if n.startswith(prefix)),
                  'jax': 'jax' in sys.modules}))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        text=True,
    )
    assert json.loads(result) == {"children": [], "jax": False}
