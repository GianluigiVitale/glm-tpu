"""Existing protected wrapper's history route/DB/archive, with no real RPCs."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from google.cloud import storage as cloud
import pytest

from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield import ws32_history_campaign as history
from scripts.greenfield import ws32_history_storage as bounded
from scripts.greenfield import ws32_history_transport as transport
from tests.greenfield.validation.test_ws32_history_transport import (
    TAG, PIN, fixture, published,
)

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts/greenfield/run_fp8_matmul_microbench.sh"


def block(marker):
    return next(part.split("\nPY\n", 1)[0]
                for part in WRAPPER.read_text().split("<<'PY'\n")[1:] if marker in part)


def account(root, db, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["accounting", str(root), PIN, str(db), str(REPO),
                                     "1", transport.protocol.KERNEL])
    exec(compile(block("run_dir, pin, db_path, repo, elapsed, expected_kernel"),
                 "<actual-history-wrapper-accounting>", "exec"), {"__name__": "__main__"})


@pytest.fixture
def composed(fixture, monkeypatch):
    result = transport.collect(**fixture.arguments)
    monkeypatch.setattr(campaign, "run_root", lambda tag: fixture.root)
    monkeypatch.setattr(history.preflight, "RUN_ROOT", fixture.root.parent)
    # Only the independently tested, expensive selected-source numerical replay
    # is substituted. Actual selected-layer routing + transport reauthentication run.
    monkeypatch.setattr(history, "_validator", lambda **kwargs: fixture.validator)
    bounded.write_json(fixture.root / "runner.json", result)
    bounded.write_bytes(fixture.root / "hlo/candidate.optimized_hlo.txt",
        (fixture.destination / "rank0/candidate_b128.optimized_hlo.txt").read_bytes())
    for name in ("runner.log", "census_pre.txt", "census_post.txt", "devices_pre.txt", "devices_post.txt"):
        bounded.write_bytes(fixture.root / name, b"CPU fixture\n")
    bounded.stream(fixture.root, "orchestrator.log", io.BytesIO(b"archive starting\n"), append=True)
    return fixture, result


def test_history_actual_selected_layer_validation_and_sqlite_nulls(composed, tmp_path, monkeypatch):
    fixture, result = composed
    db = tmp_path / "history.db"
    account(fixture.root, db, monkeypatch)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT item_id, correct, score, latency_ms, raw_output FROM items").fetchone()
        assert row[:4] == ("history_l06_8155rows_331calls_original_reproduction_v1", None, None, None)
        assert json.loads(row[4]) == result
        assert conn.execute("SELECT model_revision FROM runs").fetchone() == (
            "history-l06-two-branch-original-reproduction",)
    with sqlite3.connect(fixture.root / "results_ckpt.db") as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT count(*) FROM items").fetchone() == (1,)
    summary = json.loads((fixture.root / "summary.json").read_text())
    assert summary["claim_scope"] == history.NOTE and not summary["performance_claim"]
    projection = summary["database_projection"]
    assert projection["phase"] == "preinsert_actual"
    assert projection["raw_output_bytes"] == len(json.dumps(result, sort_keys=True).encode())
    assert projection["sql_metadata_bytes"] <= bounded.DB_SQL_METADATA_LIMIT
    assert (fixture.root / "results_ckpt.db").stat().st_size <= projection["projected_bytes"]
    assert fixture.replay_calls == ["replay", "replay"]


@pytest.mark.parametrize("field,value", [("warmup", 1), ("iterations", 1),
    ("admission_only", True), ("latency", {"p50_ms": 1}), ("performance_claim", True)])
def test_mutated_aggregate_refuses_before_db_creation(composed, tmp_path, monkeypatch, field, value):
    fixture, result = composed
    bad = deepcopy(result)
    bad[field] = value
    # Deliberately replace a test fixture to exercise admission, not the writer.
    (fixture.root / "runner.json").write_text(json.dumps(bad))
    db = tmp_path / "refused.db"
    with pytest.raises((ValueError, SystemExit)):
        account(fixture.root, db, monkeypatch)
    assert not db.exists()


def test_shell_normal_then_late_failure_archive_reuses_exact_names(composed, tmp_path, monkeypatch):
    fixture, _ = composed
    account(fixture.root, tmp_path / "history.db", monkeypatch)
    monkeypatch.setattr(cloud, "Client", fixture.bucket.client)
    monkeypatch.setattr(sys, "argv", ["archive", str(fixture.root), PIN])
    exec(compile(block("root, pin = Path(sys.argv[1]), sys.argv[2]"),
                 "<actual-history-wrapper-publication>", "exec"), {"__name__": "__main__"})
    before = deepcopy(fixture.bucket.objects)
    prefix = f"results/{TAG}/"
    terminal = json.loads(before[prefix + "SUCCESS"][1])
    assert terminal["diagnostic_only"] and terminal["boundary_diagnostic"]
    assert not terminal["admission_only"] and not terminal["performance_claim"]
    bounded.stream(fixture.root, "orchestrator.log", io.BytesIO(b"late failure\n"), append=True)
    transport.publish_controller_failure(fixture.root, client=fixture.bucket.client())
    assert all(fixture.bucket.objects[name] == original for name, original in before.items())
    assert fixture.bucket.objects[prefix + "failure_orchestrator.log"][1].endswith(b"late failure\n")
    assert not any("/diagnostic/" in name for name in fixture.bucket.objects)


def test_shell_flags_tag_floor_leases_censuses_and_bounded_sinks():
    source = WRAPPER.read_text()
    prefix = source.split('[[ $(git -C "$WORKTREE" branch --show-current)', 1)[0]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "GLM_GREENFIELD_FP8_MATMUL_KERNEL": "ws32_history_frontier"}
    env.pop("GLM_GREENFIELD_FP8_MATMUL_TAG", None)
    for name in ("GLM_GREENFIELD_FP8_MATMUL_WARMUP", "GLM_GREENFIELD_FP8_MATMUL_ITERATIONS"):
        env.pop(name, None)
    result = subprocess.run(["bash", "-c", prefix +
        '\nprintf "%s %s %s %s %s %s %s" "$TAG_STEM" "$BOUNDED_PREFILL" "$WARMUP" "$ITERATIONS" "$ROLLED_COMPILE" "$WINDOW_NUMERICAL" "$TAG"'],
        env=env, capture_output=True, text=True, timeout=20, check=True)
    values = result.stdout.split()
    assert values[:6] == ["ws32_history_frontier_l06", "1", "0", "0", "0", "0"]
    assert transport.is_tag(values[6])
    assert "exec 9>/home/gianl/glm-run/.glm_pod_workload.lock" in source
    assert "exec 8>/home/gianl/.glm-tpu-rsync.lock" in source
    assert "readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db" in source
    assert source.index("strict_census pre ||") < source.index("started=$(date +%s)")
    floor = block("dense/history diagnostic requires6GiB")
    assert "free < 6 * 1024**3" in floor
    assert 'capture_output runner.log run_runner' in source
    assert 'capture_output "devices_${label}.txt"' in source
    assert 'capture_output "census_${label}.txt"' in source
    assert "from scripts.greenfield.ws32_history_transport import publish_controller_failure" in source
    assert "from scripts.greenfield.ws32_history_transport import archive_inventory" in source
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_actual_shell_sink_captures_only_local_command_and_preserves_status(tmp_path):
    root = tmp_path / TAG
    root.mkdir()
    source = WRAPPER.read_text()
    functions = source.split("capture_output() {", 1)[1].split("\nexec 9>", 1)[0]
    script = ('set -euo pipefail\nHISTORY_FRONTIER=1\nWORKTREE=' + str(REPO) +
              '\nRUN_DIR=' + str(root) + '\ncapture_output() {' + functions +
              '\ncapture_output runner.log bash -c \'printf "local only\\n"; exit 7\'')
    result = subprocess.run(["bash", "-c", script], capture_output=True, timeout=20)
    assert result.returncode == 7
    assert (root / "runner.log").read_bytes() == b"local only\n"


@pytest.mark.parametrize("knob", ["WARMUP", "ITERATIONS", "DIAGNOSTIC_REFERENCE"])
def test_actual_zero_sampling_guard_rejects_override(knob):
    source = WRAPPER.read_text()
    start = 'if [[ $GROUPED_ADMISSION == 1 || $ROLLED_COMPILE == 1 ]]; then'
    guard = start + source.split(start)[2].split('[[ -r $RESULTS_DB', 1)[0]
    script = ('set -euo pipefail\nGROUPED_ADMISSION=1\nROLLED_COMPILE=0\n'
              'WARMUP=0\nITERATIONS=0\nDIAGNOSTIC_REFERENCE=0\n' + knob + '=1\n' + guard)
    result = subprocess.run(["bash", "-c", script], capture_output=True, timeout=20)
    assert result.returncode == 2


def test_actual_history_floor_refuses_before_mkdir(monkeypatch):
    from types import SimpleNamespace
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=(6 << 30) - 1))
    with pytest.raises(SystemExit, match="requires6GiB"):
        exec(compile(block("dense/history diagnostic requires6GiB"), "<actual-history-floor>", "exec"), {})
    source = WRAPPER.read_text()
    assert source.index("dense/history diagnostic requires6GiB") < source.index('mkdir -p "$RUN_DIR/hlo"')


@pytest.mark.parametrize("inherited", [None, "tpu"])
def test_history_accounting_explicitly_pins_cpu_without_initializing_backend(inherited):
    source = WRAPPER.read_text()
    prefix = source.split('[[ $(git -C "$WORKTREE" branch --show-current)', 1)[0]
    invocation = 'PYTHONPATH="$WORKTREE" "${BOOKKEEPING_ENV[@]}" /home/gianl/vllm-env/bin/python'
    assert source.count(invocation) == 2  # Accounting/replay and final metadata archive.
    env = {**os.environ, "GLM_GREENFIELD_FP8_MATMUL_KERNEL": "ws32_history_frontier"}
    env.pop("JAX_PLATFORMS", None)
    if inherited is not None:
        env["JAX_PLATFORMS"] = inherited
    result = subprocess.run(["bash", "-c", prefix + "\n" + invocation +
        " -c 'import os; print(os.environ[\"JAX_PLATFORMS\"])'"],
        env=env, capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout == "cpu\n"


def test_historical_recovery_extracted_census_has_no_new_helper_dependency(tmp_path):
    source = WRAPPER.read_text()
    # This is the existing recovery module's exact extraction boundary. It does
    # not include capture_output or the new HISTORY_FRONTIER variable.
    definitions = source.split("has_eight_unique_markers() {", 1)[1].split("\npost_census_done=0", 1)[0]
    script = ('set -euo pipefail\nBOUNDED_PREFILL=0\nTAG=fixture\nPOD=fixture\nZONE=fixture\n'
              'RUN_DIR=' + str(tmp_path) + '\n'
              'gcloud() { for i in {0..7}; do echo "CENSUS_OK fixture$i"; done; }\n'
              'has_eight_unique_markers() {' + definitions + '\nstrict_census post\n')
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert len((tmp_path / "census_post.txt").read_text().splitlines()) == 8


def test_actual_preinsert_projection_refuses_before_provenance_connect(composed, tmp_path, monkeypatch):
    from types import SimpleNamespace

    fixture, _ = composed
    db = tmp_path / "existing.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE untouched(value)")
    original = db.read_bytes()
    monkeypatch.setitem(bounded.FILES, "results_ckpt.db", ("database", len(original) + 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("failed DB budget reached provenance connection")

    monkeypatch.setitem(sys.modules, "provenance", SimpleNamespace(connect=forbidden))
    with pytest.raises(ValueError, match="before INSERT/launch"):
        account(fixture.root, db, monkeypatch)
    assert db.read_bytes() == original
    assert not (fixture.root / "summary.json").exists()


def test_launch_database_projection_runs_before_any_run_directory_write():
    source = WRAPPER.read_text()
    assert source.index("print(json.dumps(database_budget(Path(sys.argv[1]))") < source.index('mkdir -p "$RUN_DIR/hlo"')
    assert source.index("db_projection = history_storage.database_budget") < source.index("conn = pv.connect(db_path)")
