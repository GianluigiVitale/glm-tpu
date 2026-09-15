"""Pre-write worker limits through actual shared JSON and journal writers."""

from io import BytesIO
import json

import pytest

from scripts.greenfield import ws32_history_worker_storage as worker
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal

# Identity of the retired history campaign journal (ws32_history_execution.HistoryJournal
# at b667f00f); the storage caps under test are keyed by namespace, not by journal class.
HISTORY_PROFILE = "ws32-history-l06-db611-original-reproduction-v1"


class HistoryJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_history_numerical_journal_v1"

    def _check_identity(self, identity) -> None:
        if (identity.get("protocol") != protocol.PROTOCOL
                or identity.get("profile") != HISTORY_PROFILE
                or identity.get("compile_only") is not False
                or identity.get("diagnostic_only") is not True):
            raise ValueError("history journal requires fixed diagnostic identity")


@pytest.fixture
def root(tmp_path):
    result = tmp_path / "greenfield_fp8_ws32_history_frontier_l06_20260911T000000000000000Z" / "rank0"
    result.mkdir(parents=True)
    return result


def test_shared_json_preserves_exact_bytes_and_prior_on_refusal(root, monkeypatch):
    path = root / "runner.json"
    _atomic_json(path, {"a": 1})
    original = path.read_bytes()
    assert original == (json.dumps({"a": 1}, indent=2, sort_keys=True) + "\n").encode()
    monkeypatch.setitem(worker.METADATA, "runner.json", 100)
    with pytest.raises(ValueError, match="pre-write"):
        _atomic_json(path, {"x": "x" * 100})
    assert path.read_bytes() == original
    assert not list(root.glob("*.tmp.*"))
    _atomic_json(path, {"a": 2})
    assert json.loads(path.read_bytes()) == {"a": 2}


def test_atomic_temporary_counts_against_category(root):
    _atomic_json(root / "runner.json", {"x": "x" * (5 << 20)})
    with (root / "worker.log").open("wb") as stream:
        stream.write(b"x" * (2 << 20))
    with pytest.raises(ValueError, match="pre-write"):
        _atomic_json(root / "runner.json", {"x": "y" * ((6 << 20) - 100)})


def test_journal_append_is_bounded_before_write(root, monkeypatch):
    journal = HistoryJournal(root / "compile_journal.jsonl", dict(
        protocol=protocol.PROTOCOL, profile=HISTORY_PROFILE,
        compile_only=False, diagnostic_only=True))
    journal.phase("test")
    path = root / "compile_journal.jsonl"
    prior = path.read_bytes()
    monkeypatch.setitem(worker.METADATA, path.name, len(prior) + 1)
    with pytest.raises(ValueError, match="pre-write"):
        journal.phase("oversized")
    journal.close()
    assert path.read_bytes() == prior


def test_log_overflow_retains_prefix_and_refuses(root, monkeypatch):
    monkeypatch.setitem(worker.METADATA, "worker.log", 64 << 10)
    with pytest.raises(ValueError, match="pre-write"):
        worker.stream_log(root, BytesIO(b"x" * ((64 << 10) + 1)))
    assert (root / "worker.log").read_bytes() == b"x" * (64 << 10)
    with pytest.raises(FileExistsError):
        worker.stream_log(root, BytesIO(b""))


@pytest.mark.parametrize("name", ["unknown.json", "../runner.json"])
def test_metadata_rejects_unknown_names(root, name):
    with pytest.raises(ValueError):
        worker.write_json(root / name, {})


def test_json_symlink_and_nan_refuse(root, tmp_path):
    target = tmp_path / "old.json"
    target.write_text("original")
    (root / "runner.json").symlink_to(target)
    with pytest.raises(ValueError):
        _atomic_json(root / "runner.json", {})
    with pytest.raises(ValueError):
        _atomic_json(root / "history_report.json", {"nan": float("nan")})
    assert target.read_text() == "original"


def test_old_shared_json_path_unchanged(tmp_path):
    target = tmp_path / "old" / "runner.json"
    _atomic_json(target, {"legacy": True})
    assert json.loads(target.read_bytes()) == {"legacy": True}
