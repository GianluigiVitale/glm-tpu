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
