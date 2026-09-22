"""G4: checkpoint and format identities from code, incl. the geometry and placement hashes and all
32 owner-file header SHA-256s of the live GLM-5.3 checkpoint (copied at S0)."""
import pytest

from tools.equivalence.common import DATA, read_json

GEOMETRY = "c6ccb3f0a2abd7fe8831a93386121c0e8ee2e6b225d0201c910a54150fe63d09"
PLACEMENT = "f498b0649601585db3cf2ebb20790f47a38256bced64dd1a4c69e86ca3d0287c"


@pytest.mark.golden
@pytest.mark.golden_data("checkpoint_identity.json")
def test_checkpoint_identity(gate_check):
    gate_check("G4")


@pytest.mark.golden
@pytest.mark.golden_data("checkpoint_identity.json")
def test_baseline_matches_the_live_manifest():
    data = read_json(DATA / "checkpoint_identity.json")
    assert data["record"]["geometry"]["sha256"] == data["live"]["geometry_sha256"] == GEOMETRY
    assert data["record"]["placement"]["sha256"] == data["live"]["placement_sha256"] == PLACEMENT
    assert data["record"]["headers"] == data["live"]["headers"] and len(set(data["live"]["headers"])) == 32
    assert all(data["live_equal"].values()), data["live_equal"]
