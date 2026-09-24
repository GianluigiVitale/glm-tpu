"""G3: CPU32 execution goldens (positional leaf digests) identical to the 181c013e baseline."""

import pytest

from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("cpu_digests.json")
def test_cpu_execution_digests(gate_check):
    gate_check("G3")


@pytest.mark.golden
@pytest.mark.golden_data("cpu_digests.json")
def test_recorded_long_context_and_per_run_load():
    groups = read_json(DATA / "cpu_digests.json")["groups"]
    assert groups["donated_prompt_a"]["state_ownership"] == "exclusive_donated"
    assert groups["donated_prompt_a"]["capacity"] > 8192
    assert set(groups["load_by_run"]) == {"1536", "8704+donated", "1536#n1", "1536#n2", "1536#n3", "1536#n4"}
