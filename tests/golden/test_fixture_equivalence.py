"""Frozen fixture v1 equals the historical CPU fixture leaf for leaf (panel and non-panel)."""
import importlib.util

import pytest

from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.golden_data("fixture.json")
@pytest.mark.skipif(importlib.util.find_spec("tests.greenfield.runtime.ws32_prefill_cpu_fixture") is None,
                    reason="historical fixture archived; the S0 record in fixture.json stands")
def test_frozen_fixture_equals_historical(gate_check):
    gate_check("fixture")


@pytest.mark.golden
@pytest.mark.golden_data("fixture.json")
def test_recorded_fixture_equivalence():
    record = read_json(DATA / "fixture.json")["fixture"]
    assert set(record) == {"panel", "non_panel"}
    for value in record.values():
        assert value["identical"] and value["geometry_equal"] and value["digest"] == value["historical_digest"]
