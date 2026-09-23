"""Golden data stay compact and public-safe (digests and small summaries, versions recorded, no host
paths or run directories), the harness source carries no private infrastructure literals, and a data
file recorded on a changed production tree says why."""
import json
import re
from pathlib import Path

import pytest

from tools.equivalence.common import DATA, HARNESS_REPO
from tools.equivalence.gates import DATA_FILES, FROZEN_DATA, valid_reason

LIMIT = 256 * 1024
FORBIDDEN = re.compile(r"/home/|/dev/shm|/tmp/|optimized_request_\d|greenfield_ws32_runtime_pack_\d|gs://")
# Private site values have public-looking shapes; the harness never spells a home directory, a
# private IPv4 address, a timestamped run or pack name, or a bucket URI.
PRIVATE_SOURCE = re.compile(r"/home/|\b192\.168\.\d|\b10\.\d+\.\d+\.\d+|optimized_request_\d{8}|"
                            r"greenfield_ws32_runtime_pack_\d{8}|gs://")
FROZEN_FILES = {DATA_FILES[gate] for gate in FROZEN_DATA}
HARNESS_SOURCES = sorted(
    [*(HARNESS_REPO / "tools" / "equivalence").rglob("*.py"), *(HARNESS_REPO / "tools" / "equivalence").glob("*.md"),
     *(HARNESS_REPO / "tools" / "equivalence").glob("*.toml"), *(HARNESS_REPO / "tests" / "golden").glob("*.py")])


@pytest.mark.parametrize("path", sorted(DATA.glob("*.json")), ids=lambda p: p.name)
def test_data_file_is_compact_and_public_safe(path):
    raw = path.read_text()
    assert len(raw.encode()) <= LIMIT, f"{path.name} is {len(raw)} bytes"
    value = json.loads(raw)
    assert value["environment"]["jax"] and value["environment"]["jaxlib"], "versions must be recorded"
    assert "xla_flags" in value["environment"]
    assert not FORBIDDEN.search(raw), FORBIDDEN.search(raw).group(0)


@pytest.mark.parametrize("path", sorted(DATA.glob("*.json")), ids=lambda p: p.name)
def test_data_file_provenance(path):
    """Graph and identity goldens come from production paths equal to 181c013e. Protocol, closure,
    trace and wire goldens may be re-recorded on a changed tree only with a re-baseline marker whose
    reason is exactly one H number or stage token (rename-only re-records and H-numbered wire
    changes)."""
    value = json.loads(path.read_text())
    if path.name in FROZEN_FILES:
        assert value["source"]["production_paths_equal_baseline"] is True
    elif value["source"]["production_paths_equal_baseline"] is not True:
        reason = (value.get("rebaseline") or {}).get("reason")
        assert valid_reason(reason), f"{path.name}: changed tree without a valid re-baseline marker"


@pytest.mark.parametrize("reason", ["H3", "H16", "S1", "S1a", "S2d", "S3", "S4.2b", "S4.3", "WU-E", "WU-Docs"])
def test_rebaseline_reason_accepts_exact_tokens(reason):
    assert valid_reason(reason)


@pytest.mark.parametrize("reason", [None, "", "defaced", "cafebabe tweak", "c13e0884", "S3 moves", "fix H3",
                                    "H0", "h3", "S", "WU-", "S2d,S3"])
def test_rebaseline_reason_refuses_anything_else(reason):
    assert not valid_reason(reason)


@pytest.mark.parametrize("path", HARNESS_SOURCES, ids=lambda p: str(p.relative_to(HARNESS_REPO)))
def test_harness_source_has_no_private_literals(path: Path):
    if path.name == "test_data_contract.py":
        return  # this file spells the patterns
    match = PRIVATE_SOURCE.search(path.read_text())
    assert match is None, f"{path.relative_to(HARNESS_REPO)}: private-looking literal at offset {match.start()}"
