"""Small real local writes/SQLite snapshots; no cloud, SSH or TPU."""

import io
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import os
import signal
import sys
import time

import pytest

from scripts.greenfield import ws32_history_storage as storage

TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T200000000000000Z"


@pytest.fixture
def root(tmp_path):
    path = tmp_path / TAG
    path.mkdir()
    return path


def test_fixed_controller_allowance_and_uneven_source_owners():
    assert sum(storage.LIMITS.values()) == 469 << 20
    assert storage.LIMITS["database"] == 256 << 20
    assert storage.LIMITS["runner"] == 32 << 20
    assert storage.LIMITS["summary"] == 33 << 20
    assert sum(storage.LIMITS.values()) + storage.METADATA_RESERVE < storage.CONTROLLER_BYTES
    assert storage.FILES["source_tiles/rank4.npz"][1] == 34 << 20
    assert storage.LIMITS["sources"] == 104 << 20


def test_json_bytes_exclusive_and_append_journal(root):
    storage.write_json(root / "runner.json", {"completed": True})
    saved = (root / "runner.json").read_bytes()
    assert json.loads(saved) == {"completed": True}
    with pytest.raises(FileExistsError):
        storage.write_json(root / "runner.json", {"completed": False})
    assert (root / "runner.json").read_bytes() == saved
    tee = io.BytesIO()
    storage.stream(root, "orchestrator.log", io.BytesIO(b"a\n"), append=True, tee=tee)
    storage.stream(root, "orchestrator.log", io.BytesIO(b"b\n"), append=True)
    assert (root / "orchestrator.log").read_bytes() == b"a\nb\n"
    assert tee.getvalue() == b"a\n"


@pytest.mark.parametrize("name", ["wrong.json", "source_tiles/rank8.npz", "hlo/other.txt", "../runner.json"])
def test_unregistered_paths_refused_before_write(root, name):
    with pytest.raises(ValueError):
        storage.write_bytes(root / name, b"unsafe")
    assert list(root.iterdir()) == []


def test_symlink_and_existing_unknown_refused(root, tmp_path):
    (root / "source_tiles").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        storage.write_bytes(root / "source_tiles/rank0.npz", b"a")
    (root / "source_tiles").unlink()
    (root / "unknown").write_bytes(b"x")
    with pytest.raises(ValueError, match="unregistered existing"):
        storage.write_bytes(root / "runner.json", b"a")
    assert not (root / "runner.json").exists()


@pytest.mark.parametrize("mode", ["file", "category", "aggregate"])
def test_prewrite_limits_do_not_create_original(root, monkeypatch, mode):
    if mode == "file":
        monkeypatch.setitem(storage.FILES, "runner.json", ("runner", 1))
    elif mode == "category":
        monkeypatch.setitem(storage.LIMITS, "runner", 1)
    else:
        monkeypatch.setattr(storage, "CONTROLLER_BYTES", storage.METADATA_RESERVE + 1)
    with pytest.raises(ValueError, match="budget"):
        storage.write_bytes(root / "runner.json", b"aa")
    assert not (root / "runner.json").exists()


def test_stream_keeps_only_prior_chunks_and_source_category_is_shared(root, monkeypatch):
    monkeypatch.setattr(storage, "CHUNK_BYTES", 4)
    monkeypatch.setitem(storage.FILES, "runner.log", ("control", 6))
    with pytest.raises(ValueError, match="budget"):
        storage.stream(root, "runner.log", io.BytesIO(b"abcdefgh"))
    assert (root / "runner.log").read_bytes() == b"abcd"
    monkeypatch.setitem(storage.LIMITS, "sources", 5)
    storage.write_bytes(root / "source_tiles/rank0.npz", b"123")
    with pytest.raises(ValueError, match="budget"):
        storage.write_bytes(root / "source_tiles/rank4.npz", b"456")
    assert not (root / "source_tiles/rank4.npz").exists()


def test_nonfinite_json_and_nonjournal_append_refused(root):
    with pytest.raises(ValueError):
        storage.write_json(root / "runner.json", {"x": float("nan")})
    with pytest.raises(ValueError, match="append"):
        storage.stream(root, "runner.log", io.BytesIO(b"x"), append=True)
    assert list(root.iterdir()) == []


def test_registered_old_reference_bytes_are_separate(root):
    reference = root / "references/rank7/retained_reference"
    reference.mkdir(parents=True)
    (reference / "candidate.json").write_bytes(b"original")
    storage.write_json(root / "runner.json", {})
    assert storage._usage(root)["runner"] == (root / "runner.json").stat().st_size
    (reference / "wrong").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected retained"):
        storage.write_json(root / "summary.json", {})


def database(path, *, wal=False):
    with closing(sqlite3.connect(path)) as conn, conn:
        if wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE witness(value)")
        conn.execute("INSERT INTO witness VALUES (?)", (b"original",))


def test_real_snapshot_integrity_and_immutable_original(root, tmp_path):
    source = tmp_path / "source.db"
    database(source)
    original = source.read_bytes()
    storage.snapshot(source, root / "results_ckpt.db")
    with sqlite3.connect(root / "results_ckpt.db") as saved:
        assert saved.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert saved.execute("SELECT value FROM witness").fetchall() == [(b"original",)]
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        storage.snapshot(source, root / "results_ckpt.db")


def test_snapshot_tiny_cap_refuses_before_destination(root, tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    database(source)
    monkeypatch.setitem(storage.FILES, "results_ckpt.db", ("database", 4096))
    with pytest.raises(ValueError, match="budget"):
        storage.snapshot(source, root / "results_ckpt.db")
    assert not (root / "results_ckpt.db").exists()


def test_source_growth_after_page_count_cannot_expand_pinned_snapshot(root, tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    database(source, wal=True)
    connect = sqlite3.connect
    expected = source.stat().st_size
    monkeypatch.setitem(storage.FILES, "results_ckpt.db", ("database", expected))

    class Pinned:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def backup(self, output, **kwargs):
            # Another connection commits after count, before the first backup step.
            with connect(source) as growing:
                growing.execute("INSERT INTO witness VALUES (?)", (b"x" * (1 << 20),))
            return self.inner.backup(output, **kwargs)

    def controlled(path, *args, **kwargs):
        inner = connect(path, *args, **kwargs)
        return Pinned(inner) if str(path).endswith("?mode=ro") else inner

    monkeypatch.setattr(storage.sqlite3, "connect", controlled)
    storage.snapshot(source, root / "results_ckpt.db")
    assert (root / "results_ckpt.db").stat().st_size == expected
    with connect(root / "results_ckpt.db") as saved:
        assert saved.execute("SELECT count(*) FROM witness").fetchone() == (1,)
    with connect(source) as grown:
        assert grown.execute("SELECT count(*) FROM witness").fetchone() == (2,)


@pytest.mark.parametrize("failure", [False, True])
def test_unchanged_rpc_file_descriptor_has_prewrite_os_limit(root, monkeypatch, failure):
    output = root / "fleet_sync.log"
    monkeypatch.setitem(storage.FILES, output.name, ("control", 64))

    def unchanged():
        output.write_bytes(b"x" * (128 if failure else 32))

    if failure:
        with pytest.raises(RuntimeError, match="bounded RPC failed"):
            storage.run_logged(unchanged, output)
        assert output.stat().st_size <= 64
    else:
        storage.run_logged(unchanged, output)
        assert output.read_bytes() == b"x" * 32
    with pytest.raises(FileExistsError):
        storage.run_logged(unchanged, output)


def test_stream_cli_cpu_only(root):
    result = subprocess.run([sys.executable, "-m", "scripts.greenfield.ws32_history_storage",
                             str(root), "runner.log", "--tee"],
                            input=b"local fixture\n", capture_output=True,
                            env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=20, check=True)
    assert result.stdout == (root / "runner.log").read_bytes() == b"local fixture\n"


@pytest.mark.parametrize("ending", ["success", "failure", "timeout", "interrupt", "sigterm"])
def test_child_session_descendants_are_stopped_and_child_reaped(monkeypatch, ending):
    reader, writer = os.pipe()
    original_wait = os.waitid
    identities = []

    def action():
        child = os.getpid()
        descendant = os.fork()
        if descendant == 0:
            signal.pause()
            os._exit(0)
        os.write(writer, f"{child} {descendant}".encode())
        if ending == "failure":
            raise RuntimeError("synthetic child failure")
        if ending != "success":
            signal.pause()

    def interrupted(*args):
        identities.extend(map(int, os.read(reader, 128).split()))
        if ending == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM)
        raise KeyboardInterrupt("synthetic parent interruption")

    if ending in ("interrupt", "sigterm"):
        monkeypatch.setattr(storage.os, "waitid", interrupted)
    try:
        if ending == "success":
            storage.run_child(action)
        else:
            expected = {"failure": RuntimeError, "timeout": TimeoutError,
                        "interrupt": KeyboardInterrupt, "sigterm": InterruptedError}[ending]
            with pytest.raises(expected):
                storage.run_child(action, timeout=0.25 if ending == "timeout" else None)
        if not identities:
            identities.extend(map(int, os.read(reader, 128).split()))
        child, descendant = identities
        with pytest.raises(ChildProcessError):
            os.waitpid(child, os.WNOHANG)
        # Orphan zombies are already dead and may await the host init's reap.
        for _ in range(100):
            status = Path(f"/proc/{descendant}/stat")
            try:
                dead = status.read_text().split()[2] == "Z"
            except (FileNotFoundError, ProcessLookupError):
                dead = True
            if dead:
                break
            time.sleep(0.01)
        else:
            pytest.fail("owned descendant remains live")
    finally:
        monkeypatch.setattr(storage.os, "waitid", original_wait)
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_child_timeout_must_be_finite_positive(timeout):
    with pytest.raises(ValueError, match="timeout"):
        storage.run_child(lambda: pytest.fail("invalid timeout dispatched"), timeout=timeout)


@pytest.mark.parametrize("raw_bytes", [1, 4092, 8192, 1 << 20])
def test_database_projection_covers_actual_provenance_growth(tmp_path, raw_bytes):
    from bench import provenance as pv

    path = tmp_path / "accounting.db"
    conn = pv.connect(str(path))
    raw = "x" * raw_bytes
    metadata = dict(model="history-fixture", revision="history-l06-two-branch-original-reproduction",
                    benchmark="greenfield_fp8_ws32_history_frontier", note="untimed diagnostic")
    original = path.read_bytes()
    estimate = storage.database_budget(path, raw_output=raw, sql_metadata=metadata)
    assert path.read_bytes() == original  # Projection is read-only.
    run_id = pv.start_run(conn, model=metadata["model"], revision=metadata["revision"])
    pv.record_item(conn, run_id, benchmark=metadata["benchmark"], item_id="fixture", prompt="fixed",
                   gold=metadata["note"], raw_output=raw, correct=None, score=None, latency_ms=None)
    pv.finalize(conn, run_id, benchmark=metadata["benchmark"], metric="diagnostic_evidence_complete",
                value=1.0, note=metadata["note"])
    conn.close()
    assert path.stat().st_size <= estimate["projected_bytes"]
    assert estimate["raw_output_bytes"] == raw_bytes
    assert estimate["payload_growth_allowance"] == 2 * raw_bytes
    assert estimate["sqlite_growth_reserve"] == 1 << 20
    assert estimate["sql_metadata_bytes"] == len(json.dumps(metadata, sort_keys=True).encode())


def test_database_launch_projection_and_exact_refusal_are_read_only(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    database(source)
    original = source.read_bytes()
    launch = storage.database_budget(source)
    assert launch["phase"] == "prelaunch_maximum"
    assert launch["raw_output_bytes"] == 32 << 20
    assert launch["projected_bytes"] == len(original) + (65 << 20)
    monkeypatch.setitem(storage.FILES, "results_ckpt.db", ("database", launch["projected_bytes"] - 1))
    with pytest.raises(ValueError, match="before INSERT/launch"):
        storage.database_budget(source)
    monkeypatch.setitem(storage.FILES, "results_ckpt.db", ("database", len(original) + 1))
    with pytest.raises(ValueError, match="before INSERT/launch"):
        storage.database_budget(source, raw_output="{}", sql_metadata={})
    assert source.read_bytes() == original


@pytest.mark.parametrize("failure", ["raw", "metadata"])
def test_database_serialized_limits_refuse_before_source_read(tmp_path, monkeypatch, failure):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized serialization opened database")

    monkeypatch.setattr(storage.sqlite3, "connect", forbidden)
    monkeypatch.setitem(storage.LIMITS, "runner", 8)
    monkeypatch.setattr(storage, "DB_SQL_METADATA_LIMIT", 8)
    with pytest.raises(ValueError, match="serialization budget"):
        storage.database_budget(tmp_path / "untouched.db",
            raw_output="x" * 9 if failure == "raw" else "{}",
            sql_metadata={} if failure == "raw" else {"x": "z" * 9})
    assert not (tmp_path / "untouched.db").exists()
