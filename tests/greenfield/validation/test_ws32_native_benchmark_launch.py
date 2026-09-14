"""Original ownership and ended-publication races; no remote execution."""
from copy import deepcopy
import json
import subprocess

import pytest

from scripts.release import ws32_host_ops as launch

TAG = "greenfield_ws32_native_benchmark_20260912T220000000000000Z"
PIN = "a" * 40


def fleet():
    return [dict(rank=i,host=f"test-w-{i}",boot_id=f"boot{i}",tag=TAG,pin=PIN,
        processes=[dict(pid=100+i,start_ticks="10",argv_sha256="b"*64)],holders=[]) for i in range(8)]


def test_staggered_start_keeps_first_process_and_refuses_replacement_or_reboot():
    ready = fleet()
    early = deepcopy(ready)
    early[3]["processes"] = []
    original = launch.observe_originals(None, early)
    original = launch.observe_originals(original, ready)
    assert original == ready
    ended = deepcopy(ready)
    for row in ended: row["processes"] = []
    assert launch.observe_originals(original, ended) == ready
    replaced = deepcopy(ready)
    replaced[3]["processes"][0]["start_ticks"] = "11"
    with pytest.raises(ValueError): launch.observe_originals(original, replaced)
    reboot = deepcopy(early)
    reboot[3]["boot_id"] = "different"
    with pytest.raises(ValueError): launch.observe_originals(early, reboot)


@pytest.mark.parametrize("change", [None,"upload_pending","upload_failed","pin","exit","boot","missing","duplicate"])
def test_publication_requires_authenticated_consistent_ended_receipts(monkeypatch, change):
    observed = fleet()
    rows = deepcopy(observed)
    for r in rows:
        r["ended"] = dict(tag=TAG,code_hash=PIN,rank=r["rank"],host=r["host"],
            boot_id=r["boot_id"],worker_exit_code=0)
        r["published"] = dict(tag=TAG,code_hash=PIN,rank=r["rank"],worker_exit_code=0,publish_exit_code=0)
    if change == "upload_pending": rows[2]["published"] = None
    elif change == "upload_failed": rows[2]["published"]["publish_exit_code"] = 1
    elif change == "pin": rows[2]["published"]["code_hash"] = "c"*40
    elif change == "exit": rows[2]["published"]["worker_exit_code"] = 1
    elif change == "boot": rows[2]["ended"]["boot_id"] = "old"
    elif change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = rows[0]
    monkeypatch.setattr(launch,"ssh",lambda *a,**k:"\n".join("NATIVE_PUBLICATION "+json.dumps(r) for r in rows))
    if change in (None,"upload_pending","upload_failed"):
        result = launch.publication_state(TAG,PIN,observed)
        assert len(result) == 8
        assert (result[2]["published"] is None) == (change == "upload_pending")
    else:
        with pytest.raises(ValueError): launch.publication_state(TAG,PIN,observed)


def test_sync_command_parses_and_invalid_pin_never_reaches_shell():
    command = launch.sync_command(PIN)
    subprocess.run(["bash", "-n", "-c", command], check=True)
    assert "git reset" not in command and "tpu-vm create" not in command
    for pin in ("bad", "a" * 39, PIN + ";echo bad"):
        with pytest.raises(ValueError):
            launch.sync_command(pin)
