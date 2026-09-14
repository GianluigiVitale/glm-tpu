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
RETIRED_MODULES = {
    "m2048_association_fingerprint",
    "paired_transport",
    "transport_chain",
    "pp16_feature2_acquisition",
    "pp16_feature2_hlo",
    "pp16_feature2_loader",
    "pp16_feature2_numerical",
    "pp16_feature2_position113",
    "pp16_feature2_prefill",
    "pp16_feature2_program",
    "pp16_feature2_qkv_partial",
    "pp16_feature2_qkv_partial_event1",
    "pp16_feature2_recovery",
    "pp16_feature2_straddler",
    "pp16_feature_sharded_state",
}


def isolated_package():
    spec = importlib.util.spec_from_file_location(
        "benchmark_fixture", REPO / PATH, submodule_search_locations=[]
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_retained_original_named_export_targets_preserved():
    source = subprocess.check_output(
        ["git", "show", BASE + ":" + PATH], cwd=REPO, text=True
    )
    expected = {}
    retired_names = set()
    exports = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 1
            if node.module not in RETIRED_MODULES:
                expected.update({a.asname or a.name: node.module for a in node.names})
            else:
                retired_names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            exports = ast.literal_eval(node.value)
    package = isolated_package()
    assert package._EXPORTS == expected
    assert package.__all__ == [name for name in exports if name not in retired_names]
    for module in set(expected.values()):
        assert (REPO / PATH).with_name(module + ".py").is_file()
    for module in RETIRED_MODULES:
        assert not (REPO / PATH).with_name(module + ".py").exists()


@pytest.mark.parametrize(
    "name",
    (
        "build_m2048_strategy_nd_fingerprint",
        "build_feature2_boundary",
        "load_feature2_selective_checkpoint",
        "Feature2PrefillInputs",
        "build_transport_chain",
        "pack_paired_transport_payload",
    ),
)
def test_retired_diagnostic_alias_does_not_dispatch(monkeypatch, name):
    package = isolated_package()

    def unexpected_import(*args):
        pytest.fail("retired diagnostic must not dispatch an import")

    monkeypatch.setattr(package, "_import_module", unexpected_import)
    assert name not in dir(package)
    with pytest.raises(AttributeError):
        getattr(package, name)


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
