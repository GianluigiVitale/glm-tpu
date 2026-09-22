"""G3: CPU32 execution goldens (positional leaf digests) identical to the 181c013e baseline."""
import pytest


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("cpu_digests.json")
def test_cpu_execution_digests(gate_check):
    gate_check("G3")
