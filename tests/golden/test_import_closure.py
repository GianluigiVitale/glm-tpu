"""G6: serving-stage import closures (controller JAX-free) and the static layering scan."""
import pytest

from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.golden_data("import_closure.json")
def test_import_closures(gate_check):
    gate_check("G6")


@pytest.mark.golden
@pytest.mark.golden_data("import_closure.json")
def test_controller_never_imports_jax():
    assert read_json(DATA / "import_closure.json")["stages"]["controller"]["jax_imported"] is False
