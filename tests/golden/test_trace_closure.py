"""G7: executed-function set of the fixture production composition equals the S0 set (S1-S3)."""
import pytest


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("trace_closure.json")
def test_executed_function_set(gate_check):
    gate_check("G7")
