"""G9: request bytes and request_sha256, TokenEvent lines, worker/controller records and the
resident protocol, produced by the real code with fakes."""

import pytest

from tools.equivalence import gates, wire
from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.golden_data("wire.json")
def test_wire_formats(gate_check):
    gate_check("G9")


@pytest.mark.golden
@pytest.mark.golden_data("wire.json")
def test_request_identity_in_process():
    """The request goldens also hold in-process (no child), so a failure localizes quickly."""
    assert wire.requests_record() == gates.comparable("G9", read_json(DATA / "wire.json"))["requests"]


@pytest.mark.golden
@pytest.mark.golden_data("wire.json")
def test_recorded_refusals_are_refusals():
    """Every refusal probe of the baseline refuses, and the launcher's failure path ends in its refusal."""
    wire = read_json(DATA / "wire.json")["wire"]
    probes = {**wire["worker"]["fleet_refusals"], **wire["worker"]["main_record"]["refusals"]}
    assert probes and all(outcome != "accepted" for outcome in probes.values()), probes
    launch = wire["controller"]["launcher_main"]
    assert launch["success"]["outcome"] == "returned 0"
    assert launch["worker_failure"]["outcome"].startswith("ValueError")
    assert "cleanup_owned" in launch["worker_failure"]["remote_commands"]
