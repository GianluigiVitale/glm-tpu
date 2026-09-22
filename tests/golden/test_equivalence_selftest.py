"""G14: the normalizer's mutation self-test (invariance and sensitivity cases)."""
import pytest

from tools.equivalence.common import run_child


@pytest.mark.cpu32
@pytest.mark.slow
def test_mutation_selftest():
    result = run_child("tools.equivalence.selftest", timeout=3600, devices=1)
    failed = [c for c in result["cases"] if not c["passed"]]
    assert result["status"] == "pass" and not failed, failed
