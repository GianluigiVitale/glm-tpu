"""G9 (HTTP): the real UI/API handler with a fake resident -- status, headers (CSP), body bytes
and SSE streams, ids/timestamps normalized."""

import pytest

from tools.equivalence import gates, wire
from tools.equivalence.common import DATA, read_json


@pytest.mark.golden
@pytest.mark.golden_data("http.json")
def test_http_characterization():
    expected = gates.comparable("G9-http", read_json(DATA / "http.json"))
    actual = wire.http_record()
    assert sorted(actual) == sorted(expected)
    for name in expected:
        assert actual[name] == expected[name], name
