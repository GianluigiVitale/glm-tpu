"""Golden data stay compact and public-safe: digests and small summaries, versions recorded, no
host paths or run directories."""
import json
import re

import pytest

from tools.equivalence.common import DATA

LIMIT = 256 * 1024
FORBIDDEN = re.compile(r"/home/|/dev/shm|/tmp/|optimized_request_\d|greenfield_ws32_runtime_pack_\d|gs://")


@pytest.mark.parametrize("path", sorted(DATA.glob("*.json")), ids=lambda p: p.name)
def test_data_file_is_compact_and_public_safe(path):
    raw = path.read_text()
    assert len(raw.encode()) <= LIMIT, f"{path.name} is {len(raw)} bytes"
    value = json.loads(raw)
    assert value["environment"]["jax"] and value["environment"]["jaxlib"], "versions must be recorded"
    assert "xla_flags" in value["environment"]
    assert value["source"]["production_paths_equal_baseline"] is True
    assert not FORBIDDEN.search(raw), FORBIDDEN.search(raw).group(0)
