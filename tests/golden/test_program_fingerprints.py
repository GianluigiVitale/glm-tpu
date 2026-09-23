"""G1 (fixture tier, every commit) and G2 (production tier, stage exits): normalized TPU StableHLO
digests, N8 signatures and compile arguments of every run's programs identical to the 181c013e
baseline; G1-protocol/G2-protocol: the load protocol and production defaults (characterization,
re-baselined only with a reviewed reason)."""
import os

import pytest


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.golden_data("fingerprints_fixture.json")
def test_fixture_tier_fingerprints(gate_check):
    gate_check("G1")


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.golden_data("load_protocol_fixture.json")
def test_fixture_load_protocol(gate_check):
    gate_check("G1-protocol")


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("fingerprints_production.json")
@pytest.mark.skipif(os.environ.get("GLM_EQUIVALENCE_PRODUCTION") != "1",
                    reason="production tier (78 layers, ~10 min, several GB RAM): set GLM_EQUIVALENCE_PRODUCTION=1 "
                           "or run `python -m tools.equivalence check --tier production`")
def test_production_tier_fingerprints(gate_check):
    gate_check("G2")


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("load_protocol_production.json")
@pytest.mark.skipif(os.environ.get("GLM_EQUIVALENCE_PRODUCTION") != "1",
                    reason="production tier: set GLM_EQUIVALENCE_PRODUCTION=1")
def test_production_load_protocol(gate_check):
    gate_check("G2-protocol")
