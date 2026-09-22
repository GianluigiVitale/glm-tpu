"""G1 (fixture tier, every commit) and G2 (production tier, stage exits): normalized TPU StableHLO
digests and N8 signatures identical to the 181c013e baseline."""
import os

import pytest


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.golden_data("fingerprints_fixture.json")
def test_fixture_tier_fingerprints(gate_check):
    gate_check("G1")


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("fingerprints_production.json")
@pytest.mark.skipif(os.environ.get("GLM_EQUIVALENCE_PRODUCTION") != "1",
                    reason="production tier (78 layers, ~10 min, several GB RAM): set GLM_EQUIVALENCE_PRODUCTION=1 "
                           "or run `python -m tools.equivalence check --tier production`")
def test_production_tier_fingerprints(gate_check):
    gate_check("G2")
