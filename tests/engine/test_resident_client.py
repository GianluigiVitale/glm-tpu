"""Tests of how :class:`glm_tpu.engine.resident_client.Resident` finds and authenticates its controller (H18): from
the run directory's own controller records, which ``glm-tpu ask --keep-loaded`` writes as the controller module does,
or from an operator's dispatch receipt naming that same controller. A real child process stands in for the
controller (its command line names no fleet module, so no heavy gate sees a live run)."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from glm_tpu.engine import request
from glm_tpu.engine import resident_client
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.engine.resident_client import Resident, controller_receipt, resident_command
from glm_tpu.utils.json_utils import canonical

PIN = "a" * 40


def start_ticks(pid: int) -> str:
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


@pytest.fixture
def controller():
    """A live process whose command line is that of ``python -m glm_tpu ask "..." --keep-loaded``."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)", "ask", "Q", "--keep-loaded"])
    try:
        for _ in range(100):  # until the exec'd command line is visible
            if b"--keep-loaded" in (Path("/proc") / str(process.pid) / "cmdline").read_bytes():
                break
            time.sleep(0.05)
        yield process
    finally:
        process.kill()
        process.wait()


def resident_run(tmp_path: Path, pid: int, ticks: str, *, measured: bool = True) -> Path:
    """A resident run directory as the controller writes it: controller_identity.json, the staged request.json and,
    after the first answer, resident-measurement.json."""
    run = tmp_path / ("optimized_request_" + "20000101T" + "0" * 12 + "Z")  # a synthetic run name
    run.mkdir(mode=0o700)
    raw = canonical(request.from_token_ids([7], request_id="r0", max_new_tokens=2, context_capacity=32768)) + b"\n"
    identity = dict(
        code_hash=PIN,
        controller_pid=pid,
        controller_start_ticks=ticks,
        request_file_sha256=sha256(raw).hexdigest(),
    )
    for name, data in (
        (protocol.CONTROLLER_IDENTITY_FILE, json.dumps(identity).encode()),
        (resident_client.STAGED_REQUEST, raw),
        *([(protocol.MEASUREMENT_FILE, json.dumps(dict(code_hash=PIN)).encode())] if measured else []),
    ):
        (run / name).write_bytes(data)
        (run / name).chmod(0o600)
    (run / protocol.INBOX_DIR).mkdir(mode=0o700)
    return run


def attach(monkeypatch, run: Path, dispatch: Path | None = None) -> Resident:
    """``Resident(run, dispatch, ...)`` without the tokenizer and template files (faked)."""
    from glm_tpu.config import model
    import transformers

    monkeypatch.setattr(model, "verified_template", lambda repo, path: "template")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: "tokenizer")
    return Resident(run, dispatch, Path("/repo"), Path("/model"))


@pytest.mark.parametrize(
    "argv,resident",
    [
        ([b"/venv/bin/python", b"-m", b"glm_tpu", b"ask", b"Q", b"--keep-loaded", b""], True),
        ([b"/venv/bin/python", b"/venv/bin/glm-tpu", b"ask", b"--questions", b"q.json", b"--keep-loaded", b""], True),
        ([b"python", b"-m", protocol.CONTROLLER_MODULE.encode(), b"--request", b"r.json", b"--keep-loaded", b""], True),
        ([b"python", b"-m", b"glm_tpu", b"ask", b"Q", b""], False),
        ([b"python", b"-m", protocol.CONTROLLER_MODULE.encode(), b"--request", b"r.json", b""], False),
        ([b"python", b"server.py", b"--keep-loaded", b""], False),
    ],
    ids=["ask module", "ask script", "controller module", "ask not resident", "controller not resident", "other"],
)
def test_resident_command(argv, resident):
    assert resident_command(argv) is resident


def test_the_run_names_its_controller(tmp_path: Path):
    run = resident_run(tmp_path, 1234, "99")
    assert controller_receipt(run) == dict(
        pid=1234, start_ticks="99", code_hash=PIN, request=str(run / resident_client.STAGED_REQUEST)
    )


def test_a_staged_request_that_differs_from_the_record_is_refused(tmp_path: Path):
    run = resident_run(tmp_path, 1234, "99")
    (run / resident_client.STAGED_REQUEST).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="staged request differs"):
        controller_receipt(run)


def test_a_controller_record_that_is_not_owner_only_is_refused(tmp_path: Path):
    run = resident_run(tmp_path, 1234, "99")
    (run / protocol.CONTROLLER_IDENTITY_FILE).chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        controller_receipt(run)


def test_an_ask_session_attaches_without_a_receipt(tmp_path: Path, monkeypatch, controller):
    run = resident_run(tmp_path, controller.pid, start_ticks(controller.pid))
    backend = attach(monkeypatch, run)
    assert backend.capacity == 32768 and backend.identity["pid"] == controller.pid
    backend.check()  # every later step authenticates the live controller again
    (run / "inbox" / "stop.json").write_text('{"stop":true}')
    with pytest.raises(ValueError, match="resident model is unavailable"):
        backend.check()


def test_a_dispatch_receipt_must_name_the_runs_controller(tmp_path: Path, monkeypatch, controller):
    ticks = start_ticks(controller.pid)
    run = resident_run(tmp_path, controller.pid, ticks)
    receipt = tmp_path / "dispatch.json"
    command = [a.decode() for a in (Path("/proc") / str(controller.pid) / "cmdline").read_bytes().split(b"\0") if a]
    command += ["--request", str(run / resident_client.STAGED_REQUEST)]
    receipt.write_text(json.dumps(dict(pid=controller.pid, start_ticks=ticks, command=command, code_hash=PIN)))
    assert attach(monkeypatch, run, receipt).capacity == 32768
    receipt.write_text(json.dumps(dict(pid=os.getpid(), start_ticks=ticks, command=command, code_hash=PIN)))
    with pytest.raises(ValueError, match="names another controller"):
        attach(monkeypatch, run, receipt)


def test_the_server_waits_for_the_first_answer(tmp_path: Path, monkeypatch, controller):
    run = resident_run(tmp_path, controller.pid, start_ticks(controller.pid), measured=False)
    with pytest.raises(ValueError, match="start the server after the controller printed RESIDENT_RESULT"):
        attach(monkeypatch, run)


@pytest.mark.parametrize("change", ["ticks", "ended", "not resident"])
def test_a_controller_that_is_not_the_recorded_one_is_unavailable(tmp_path: Path, monkeypatch, controller, change):
    ticks = start_ticks(controller.pid)
    if change == "ticks":
        run = resident_run(tmp_path, controller.pid, str(int(ticks) + 1))
    elif change == "ended":
        controller.kill()
        controller.wait()
        run = resident_run(tmp_path, controller.pid, ticks)
    else:
        run = resident_run(tmp_path, os.getpid(), start_ticks(os.getpid()))  # pytest: not a resident command line
    with pytest.raises(ValueError, match="resident model is unavailable"):
        attach(monkeypatch, run)


def test_a_controller_of_another_user_is_unavailable(tmp_path: Path, monkeypatch, controller):
    run = resident_run(tmp_path, controller.pid, start_ticks(controller.pid))
    backend = attach(monkeypatch, run)
    real = resident_client._process
    monkeypatch.setattr(resident_client, "_process", lambda pid: (*real(pid)[:3], os.geteuid() + 1))
    with pytest.raises(ValueError, match="resident model is unavailable"):
        backend.check()
