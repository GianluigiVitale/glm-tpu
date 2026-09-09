"""Actual probe/entry and nine-call writer lifecycle, fixture runtime/compute."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import probe_ws32_prefill_layer as probe
from scripts.greenfield import run_short_decoder_ws32 as runner
from scripts.greenfield import ws32_dense_frontier_entry as entry
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from tests.greenfield.validation.test_ws32_dense_frontier_execution import staged

TAG = "greenfield_fp8_ws32_dense_frontier_d01_20260909T150000000000000Z"
PIN = "a" * 40
REPO = Path(__file__).resolve().parents[3]


def preflight(root, *, hostname=None):
    (root / "retained_preflight.json").write_text(
        json.dumps(
            dict(
                tag=TAG,
                code_hash=PIN,
                launch_rank=0,
                hostname=entry.socket.gethostname() if hostname is None else hostname,
                protocol=protocol.PROTOCOL,
            )
        )
    )


@pytest.mark.parametrize("norm_mode", [False, True, "canonical"])
def test_probe_dense_route_precedes_historical_layer_selection(
    tmp_path, monkeypatch, norm_mode
):
    tag = TAG
    if norm_mode == "canonical":
        from tests.greenfield.validation.test_ws32_dense_canonical_entry_transport import TAG as tag
    elif norm_mode:
        from tests.greenfield.validation.test_ws32_dense_norm_entry_transport import (
            TAG as tag,
        )
    args = NS(
        expected_code_hash=PIN,
        coordinator_address="127.0.0.1:8476",
        process_id=0,
        output_dir=tmp_path / tag / "rank0",
    )
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", tag)
    monkeypatch.setattr(probe.argparse.ArgumentParser, "parse_args", lambda _: args)
    monkeypatch.setattr(probe, "_git_head", lambda: PIN)
    monkeypatch.setattr(
        probe, "Path", lambda s: tmp_path if s == "/home/gianl/glm-run" else Path(s)
    )

    def forbidden(*a, **k):
        raise AssertionError("dense route entered historical layer picker")

    monkeypatch.setattr(probe, "layer_from_tag", forbidden)

    def execute(actual, *, tag, repo):
        assert actual is args and tag == args.output_dir.parent.name and repo == REPO
        assert args.num_processes == 8 and args.slice_name == "db-v4-64-od"
        assert (
            args.topology_capture_root,
            args.topology_sha256,
            args.topology_fleet_sha256,
            args.mesh_sha256,
        ) == (probe.TOPOLOGY, probe.TOPOLOGY_SHA, probe.FLEET_SHA, probe.MESH_SHA)
        return 0

    monkeypatch.setattr(entry, "execute", execute)
    assert probe.main() == 0
    (args.output_dir / "runner.json").write_text("existing")
    with pytest.raises(FileExistsError):
        probe.main()


@pytest.mark.parametrize(
    "failure", [None, "reproduction", "finalize", "terminal_peer", "terminal_write"]
)
def test_entry_actual_nine_calls_and_voted_terminal(tmp_path, monkeypatch, failure):
    run, actual_record, events = staged(
        tmp_path,
        monkeypatch,
        failure if failure in ("reproduction", "finalize") else None,
    )
    preflight(tmp_path)
    monkeypatch.setattr(
        entry.model_admission, "require_acquired_model_source", lambda *a, **k: None
    )
    runtime = object()
    monkeypatch.setattr(runner, "_initialize_runtime", lambda args: runtime)

    def execute_bound(**kwargs):
        assert (
            kwargs["runtime"] is runtime
            and kwargs["inspect_program"] is entry.admission.inspect_program
        )
        try:
            run()
        finally:
            kwargs["record"].update(actual_record)

    monkeypatch.setattr(entry.runtime_module, "execute_bound", execute_bound)
    from jax.experimental import multihost_utils

    votes = []

    def vote(passed):
        votes.append(bool(passed))
        return np.asarray([False] if failure == "terminal_peer" else [passed])

    monkeypatch.setattr(multihost_utils, "process_allgather", vote)
    if failure == "terminal_write":
        from scripts.greenfield import prefill_window_acquisition as lifecycle

        original = lifecycle._atomic_json

        def write(path, record):
            if record.get("status") == entry.STATUS:
                raise OSError("fixture terminal publication failure")
            return original(path, record)

        monkeypatch.setattr(lifecycle, "_atomic_json", write)
    args = NS(expected_code_hash=PIN, process_id=0, output_dir=tmp_path)
    if failure:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            entry.execute(args, tag=TAG, repo=REPO)
    else:
        assert entry.execute(args, tag=TAG, repo=REPO) == 0
    saved = json.loads((tmp_path / "runner.json").read_text())
    assert saved["status"] == ("DIAGNOSTIC_FAILED" if failure else entry.STATUS)
    assert saved["performance_claim"] is False and saved["numerical_promotion"] is False
    assert len(saved["call_evidence"]) == 9
    assert (tmp_path / "wide_final.npz").exists() and (
        tmp_path / "narrow_128.npz"
    ).exists()
    if failure not in ("reproduction", "finalize"):
        assert votes == [failure != "terminal_write"]


@pytest.mark.parametrize("failure", ["host", "source", "initialize"])
def test_preinit_failure_record_without_any_model_dispatch(
    tmp_path, monkeypatch, failure
):
    preflight(tmp_path, hostname="wrong" if failure == "host" else None)
    initialized = []

    def source(*a, **k):
        if failure == "source":
            raise ValueError("fixture source drift")

    def initialize(args):
        initialized.append(True)
        raise ValueError("fixture runtime failure")

    monkeypatch.setattr(entry.model_admission, "require_acquired_model_source", source)
    monkeypatch.setattr(runner, "_initialize_runtime", initialize)

    def forbidden(**kwargs):
        raise AssertionError("unexpected model dispatch")

    monkeypatch.setattr(entry.runtime_module, "execute_bound", forbidden)
    with pytest.raises(ValueError):
        entry.execute(
            NS(expected_code_hash=PIN, process_id=0, output_dir=tmp_path),
            tag=TAG,
            repo=REPO,
        )
    assert initialized == ([True] if failure == "initialize" else [])
    record = json.loads((tmp_path / "runner.json").read_text())
    assert record["status"] == "DIAGNOSTIC_FAILED" and record["programs"] == {}
