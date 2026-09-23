"""G6: serving-stage import closures (controller JAX-free) and the static layering scan.

``test_import_closures`` runs every stage (the graph stage builds the real runtime on 32 CPU devices:
heavy, refused while a TPU run is live); ``test_import_closures_static`` is the light subset
(entry-module stages, the launcher exercise and the static scan)."""
import pytest

from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.golden_data("import_closure.json")
def test_import_closures(gate_check):
    gate_check("G6")


@pytest.mark.golden
@pytest.mark.golden_data("import_closure.json")
def test_import_closures_static(gate_check):
    gate_check("G6-static")


@pytest.mark.golden
@pytest.mark.golden_data("import_closure.json")
def test_controller_never_imports_jax():
    assert read_json(DATA / "import_closure.json")["stages"]["controller"]["jax_imported"] is False
