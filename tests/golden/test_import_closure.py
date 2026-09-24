"""G6: serving-stage import closures (controller JAX-free) and the static layering scan.

``test_import_closures`` runs every stage (the graph stage builds the real runtime on 32 CPU devices:
heavy, refused while a TPU run is live); ``test_import_closures_static`` is the light subset
(entry-module stages, the launcher exercise and the static scan)."""
import pytest

from tools.equivalence.common import DATA, read_json
from tools.equivalence.import_closure import lean_violations


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


@pytest.mark.golden
@pytest.mark.golden_data("import_closure.json")
def test_worker_and_graph_closures_are_lean():
    """S2f: the worker preflight, worker main and graph closures hold only glm_tpu modules (the
    process entry modules are glm_tpu modules since S3; G6 enforces it on every fresh record)."""
    assert lean_violations(read_json(DATA / "import_closure.json")["stages"]) == []


def test_lean_allowlist_flags_every_module_outside_glm_tpu():
    stages = dict(worker_main=dict(modules=["glm_tpu", "glm_tpu.runner.tpu_runner", "glm_tpu.worker.tpu_worker",
                                            "bench.x", "glm_tpu_extra.y"]),
                  graph=dict(modules=["tools.x", "glm_tpu.runner.programs"]),
                  serving=dict(modules=["tests.fixtures.site"]))  # serving is not lean-checked
    assert lean_violations(stages) == ["stages.worker_main.not_lean+bench.x",
                                       "stages.worker_main.not_lean+glm_tpu_extra.y",
                                       "stages.graph.not_lean+tools.x"]
