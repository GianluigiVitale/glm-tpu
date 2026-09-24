"""G1 (fixture tier, every commit) and G2 (production tier, stage exits): normalized TPU StableHLO
digests, N8 signatures and compile arguments of every run's programs identical to the 181c013e
baseline; G1-protocol/G2-protocol: the load protocol and production defaults (characterization,
re-baselined only with a reviewed reason)."""

import os

import pytest

from tools.equivalence.common import DATA, read_json


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
@pytest.mark.skipif(
    os.environ.get("GLM_EQUIVALENCE_PRODUCTION") != "1",
    reason="production tier (78 layers, ~10 min, several GB RAM): set GLM_EQUIVALENCE_PRODUCTION=1 "
    "or run `python -m tools.equivalence check --tier production`",
)
def test_production_tier_fingerprints(gate_check):
    gate_check("G2")


@pytest.mark.golden
@pytest.mark.cpu32
@pytest.mark.slow
@pytest.mark.golden_data("load_protocol_production.json")
@pytest.mark.skipif(
    os.environ.get("GLM_EQUIVALENCE_PRODUCTION") != "1", reason="production tier: set GLM_EQUIVALENCE_PRODUCTION=1"
)
def test_production_load_protocol(gate_check):
    gate_check("G2-protocol")


@pytest.mark.golden
@pytest.mark.parametrize("name", ["fingerprints_fixture.json", "fingerprints_production.json"])
def test_recorded_safety_facts_are_the_safe_ones(name):
    """The frozen safety record holds refusals and verification where they belong (a baseline
    recorded from a broken tree would otherwise freeze the breakage)."""
    safety = read_json(DATA / name)["safety"]
    assert safety["runs"], "no runs recorded"
    for key, run in safety["runs"].items():
        assert run["graph_consensus_probe"].startswith("refused"), key
        kwargs = run["verify_checkpoint"]["kwargs"]
        assert kwargs["verify_file_hashes"] is True and kwargs["local_slot_layout"] is True, key
        assert len(kwargs["verify_file_hash_slots"]) == 4, key
        assert {"decode", "prefill_114", "prefill_128"} & set(run["hlo_admitted_programs"]), key
        assert run["graph_consensus_calls"] >= 6, key
    memory, hlo = safety["admission"]["memory"], safety["admission"]["hlo"]
    assert memory["one_byte_below_limit"] == "fits" and memory["at_limit"] == "does not fit"
    assert hlo.pop("physical_axes_pod_4096_bytes") == "accepted"
    assert hlo and all(verdict.startswith("refused") for verdict in hlo.values()), hlo
