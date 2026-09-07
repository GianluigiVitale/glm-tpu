"""A lost observer must never make a live numerical run appear complete."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/greenfield/watch_ws32_run.py"
spec = importlib.util.spec_from_file_location("watch_ws32_run", SCRIPT)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
TAG = "greenfield_ws32_short_decoder_128k_d0_0_numerical_20260907T064941550123130Z"
PIN = "a" * 40


def fleet():
    return [dict(rank=r, host=f"pod-w-{r}", boot_id=f"boot-{r}", tag=TAG, pin=PIN,
                 processes=[dict(pid=100 + r, start_ticks="1000", parent_pid=90 + r,
                                 executable="python3", argv_sha256="b" * 64)],
                 holders=[100 + r], progress="chunk 7", output_present=False)
            for r in range(8)]


def encode(rows):
    return "ssh progress\n" + "\n".join(watch.PREFIX + json.dumps(row) for row in rows)


def test_parse_complete_fleet_ignores_ssh_noise():
    assert watch.parse_fleet(encode(list(reversed(fleet()))), TAG, PIN) == fleet()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "pin", "tag", "host", "extra_process"])
def test_parse_refuses_incomplete_or_wrong_identity(mutation):
    rows = fleet()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "extra_process":
        rows[0]["processes"] *= 2
    else:
        rows[0][mutation] = "wrong"
    with pytest.raises(ValueError):
        watch.parse_fleet(encode(rows), TAG, PIN)


@pytest.mark.parametrize("field", ["pid", "start_ticks", "argv_sha256", "executable"])
def test_original_identity_refuses_replacement(field):
    observed = fleet()
    observed[0]["processes"][0][field] = "changed"
    with pytest.raises(ValueError, match="replaced"):
        watch.require_original_fleet(fleet(), observed)


def test_reboot_is_not_terminal():
    observed = fleet()
    observed[0].update(boot_id="new", processes=[], holders=[])
    with pytest.raises(ValueError, match="boot_id"):
        watch.require_original_fleet(fleet(), observed)


def test_resume_preserves_original_process_baseline_after_partial_completion():
    first = dict(status="OBSERVED", tag=TAG, pin=PIN, fleet=fleet())
    later = deepcopy(first)
    later["fleet"][0].update(processes=[], holders=[])
    receipt = json.dumps(first) + "\n" + json.dumps(later) + "\n"
    assert watch.resume_baseline(receipt, TAG, PIN) == fleet()
    changed = deepcopy(later)
    changed["fleet"][1]["processes"][0]["start_ticks"] = "9999"
    for broken in (receipt[:-1], receipt.replace(PIN, "c" * 40),
                   receipt + json.dumps(changed) + "\n"):
        with pytest.raises(ValueError):
            watch.resume_baseline(broken, TAG, PIN)


def test_poll_retains_leases_on_timeout_and_requires_two_idle_observations(tmp_path, monkeypatch):
    import fcntl
    rows = fleet()
    busy = deepcopy(rows)
    for row in busy:
        row["processes"] = []
    idle = deepcopy(busy)
    for row in idle:
        row["holders"] = []
    sequence = iter([rows, subprocess.TimeoutExpired("ssh", 55), busy, idle, idle])
    locks = (tmp_path / "workload.lock", tmp_path / "rsync.lock")
    monkeypatch.setattr(watch, "LOCKS", locks)
    monkeypatch.setattr(watch, "RUN_ROOT", tmp_path)
    (tmp_path / TAG).mkdir()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--tag", TAG, "--code-hash", PIN])

    def observe(*_):
        for lock in locks:
            with lock.open("a") as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = next(sequence)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(watch, "observe", observe)
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    assert watch.main() == 0
    receipts = [json.loads(line) for line in (tmp_path / TAG / "watch.jsonl").read_text().splitlines()]
    assert [r["status"] for r in receipts] == [
        "OBSERVED", "UNKNOWN_RETRYING", "OBSERVED", "OBSERVED", "READY_FOR_CENSUS"]
    assert receipts[-1]["numerical_success_claim"] is False
    for lock in locks:
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A second invocation resumes the original baseline even though all workers
    # have now exited. It must not require eight live processes again.
    monkeypatch.setattr(watch, "observe", lambda *_: idle)
    assert watch.main() == 0
