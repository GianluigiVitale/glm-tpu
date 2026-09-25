"""The harness's permanent recorded-name tables. No re-baseline clears them, because the records that
read them are frozen or keep the 181c013e spellings:

* ``driver.RECORDED_NAMES``: the admission functions whose verdicts the frozen G1/G2 ``safety``
  record holds (``verdicts.admission_cases``) and under whose name ``compile``'s HLO admission is
  faked (``project_memory`` and ``check_hlo_collectives`` since WU-R). Negative control: after a
  rename and a cleared ``closure_map.toml``, a stale or missing row makes a verdict read ``<absent>``,
  which differs from the frozen record (G1/G2 would fail).
* ``driver.HOMES``: the faked loader functions, keyed by the recorded stub names that the load
  protocol and the frozen ``verify_checkpoint`` fact read.
* ``kernel_renames.toml`` ``[names]``: one row per public Pallas kernel name.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest

from tools.equivalence import closure_map, driver, verdicts
from tools.equivalence.common import DATA, read_json
from tools.equivalence.lowering import kernel_renames

FROZEN = ("fingerprints_fixture.json", "fingerprints_production.json")
LABEL = {"memory_projection": "memory", "inspect_research_hlo": "hlo"}  # labels of admission_cases()


def _frozen(name: str) -> dict:
    return read_json(DATA / name)["safety"]["admission"]


def _literal_calls(module: object, function: str) -> set[str]:
    """String literals passed as the first argument of ``function(...)`` calls in ``module``'s source."""
    tree = ast.parse(inspect.getsource(module))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _verdicts() -> dict:
    return verdicts.safety_verdicts(verdicts.admission_cases())


@pytest.fixture
def cleared(monkeypatch):
    """``closure_map.toml`` as after a re-baseline (empty); the admission homes are loaded."""
    monkeypatch.setattr(closure_map, "load", lambda path=closure_map.MAP: closure_map.ClosureMap())
    for name in driver.ADMISSION_HOMES:
        importlib.import_module(name)


def _rename(monkeypatch: pytest.MonkeyPatch, recorded: str) -> tuple[str, object]:
    """Rename the current definition of ``recorded`` in its home and in every admission home that
    binds it (the runtime module's import), as a work unit's rename would; return the new name and
    the function."""
    function = driver._definition(recorded)
    assert callable(function), function
    current, new = driver.RECORDED_NAMES[recorded], driver.RECORDED_NAMES[recorded] + "_renamed_by_test"
    for name in {function.__module__, *driver.ADMISSION_HOMES}:
        if getattr(sys.modules[name], current, None) is function:
            monkeypatch.delattr(sys.modules[name], current)
            monkeypatch.setattr(sys.modules[name], new, function, raising=False)
    return new, function


def test_the_tables_cover_every_name_the_frozen_record_reads():
    assert set(driver.RECORDED_NAMES) == set(LABEL) == _literal_calls(verdicts, "_definition")
    assert _literal_calls(driver, "current_names") <= set(driver.RECORDED_NAMES)
    assert all(value.isidentifier() for value in driver.RECORDED_NAMES.values())
    assert "verify_ws32_runtime_checkpoint" in driver.HOMES  # verdicts.run_safety's verify_checkpoint


@pytest.mark.parametrize("recorded", sorted(driver.RECORDED_NAMES))
def test_each_row_resolves_to_exactly_one_definition(cleared, recorded):
    function = driver._definition(recorded)
    assert callable(function), function  # not <absent> or <ambiguous: n definitions>
    assert function.__name__ == driver.RECORDED_NAMES[recorded]


@pytest.mark.parametrize("name", FROZEN)
def test_admission_verdicts_found_through_the_table_equal_the_frozen_record(cleared, name):
    assert _verdicts() == _frozen(name)


@pytest.mark.parametrize("recorded", sorted(driver.RECORDED_NAMES))
def test_a_rename_is_found_only_through_its_row(cleared, monkeypatch, recorded):
    frozen = _frozen(FROZEN[0])
    new, function = _rename(monkeypatch, recorded)
    # The renaming commit: its closure_map.toml [functions] entry finds the function.
    entry = closure_map.ClosureMap(functions={f"{function.__module__}:{new}": f"{function.__module__}:{recorded}"})
    with monkeypatch.context() as scoped:
        scoped.setattr(closure_map, "load", lambda path=closure_map.MAP: entry)
        assert _verdicts() == frozen
    # After the table is cleared: a stale row, or none, loses it; the frozen comparison fails.
    stale = _verdicts()
    assert stale[LABEL[recorded]] == driver.ABSENT and stale != frozen
    monkeypatch.delitem(driver.RECORDED_NAMES, recorded)
    missing = _verdicts()
    assert missing[LABEL[recorded]] == driver.ABSENT and missing != frozen
    # The row updated in the renaming commit keeps the frozen verdicts.
    monkeypatch.setitem(driver.RECORDED_NAMES, recorded, new)
    assert _verdicts() == frozen


def test_the_hlo_admission_fake_follows_the_table(cleared, monkeypatch):
    recorded = "inspect_research_hlo"
    new, real = _rename(monkeypatch, recorded)
    homes = [sys.modules[name] for name in driver.ADMISSION_HOMES if getattr(sys.modules[name], new, None) is real]
    assert sys.modules[real.__module__] in homes and sys.modules[driver.RUNTIME_MODULE] in homes
    compiler = driver.ProductionCompile(SimpleNamespace(), fingerprint=False, keep=None)
    monkeypatch.setitem(driver.RECORDED_NAMES, recorded, new)
    with compiler.patches():
        assert all(getattr(module, new) is not real for module in homes)
    assert all(getattr(module, new) is real for module in homes)
    # Without the row the fake misses the renamed parser; the real one refuses the stand-in text.
    monkeypatch.delitem(driver.RECORDED_NAMES, recorded)
    with compiler.patches():
        assert all(getattr(module, new) is real for module in homes)
    with pytest.raises(ValueError):
        real(driver.STANDIN_HLO.format(name="decode"))


def test_the_hlo_admission_fake_follows_a_closure_map_entry(cleared, monkeypatch):
    """In the renaming commit the ``[functions]`` entry alone (row not yet updated) also moves the fake."""
    recorded = "inspect_research_hlo"
    new, real = _rename(monkeypatch, recorded)
    homes = [sys.modules[name] for name in driver.ADMISSION_HOMES if getattr(sys.modules[name], new, None) is real]
    assert sys.modules[real.__module__] in homes and sys.modules[driver.RUNTIME_MODULE] in homes
    compiler = driver.ProductionCompile(SimpleNamespace(), fingerprint=False, keep=None)
    entry = closure_map.ClosureMap(functions={f"{real.__module__}:{new}": f"{real.__module__}:{recorded}"})
    with monkeypatch.context() as scoped:
        scoped.setattr(closure_map, "load", lambda path=closure_map.MAP: entry)
        with compiler.patches():
            assert all(getattr(module, new) is not real for module in homes)
    assert all(getattr(module, new) is real for module in homes)
    # The table cleared and the row stale: the real parser stays in place (and refuses the stand-in text).
    with compiler.patches():
        assert all(getattr(module, new) is real for module in homes)


@pytest.mark.parametrize("stub", sorted(driver.HOMES))
def test_each_loader_stub_has_exactly_one_current_home(stub):
    assert callable(getattr(driver.LoadStubs, stub, None))  # the recorded key of the load protocol
    defined = []
    for module_name, attribute in driver.HOMES[stub]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(getattr(module, attribute, None), "__module__", None) == module_name:
            defined.append(f"{module_name}:{attribute}")
    assert len(defined) == 1, defined


def test_kernel_renames_has_one_row_per_public_kernel_name():
    table = kernel_renames()
    public = getattr(importlib.import_module(table["module"]), table["attribute"])
    assert (table["module"], table["attribute"]) == ("glm_tpu.kernels.names", "KERNEL_NAMES")
    assert sorted(table["names"]) == sorted(public.values())
    assert len(set(table["names"].values())) == len(table["names"])  # distinct 181c013e spellings
