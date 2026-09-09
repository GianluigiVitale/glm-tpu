"""Actual recovery/accounting/archive composition; cloud/census/math fixtures.

No TPU, real bucket, canonical DB or process lock is touched by this test.
"""

import builtins
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from google.cloud import storage
import pytest

from scripts.greenfield import recover_prefill_phase as recovery
from scripts.greenfield import ws32_dense_frontier_transport as transport
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.validation.test_ws32_dense_frontier_transport import (
    published,
    TAG,
    PIN,
)

REPO = Path(__file__).resolve().parents[3]


def test_existing_dense_originals_recover_without_launch_or_reupload(
    tmp_path, monkeypatch
):
    root = tmp_path / TAG
    root.mkdir()
    bucket = published(tmp_path)
    monkeypatch.setattr(storage, "Client", bucket.client)
    monkeypatch.setattr(campaign, "run_root", lambda tag: root)
    # Explicit numerical replay seam. Actual full-original replay is separate.
    monkeypatch.setattr(
        transport.evidence,
        "validate_fleet",
        lambda *a, **kw: dict(reproduced=True, owners=32, hosts=8),
    )
    transport.collect(
        tag=TAG,
        pin=PIN,
        root=root / "fleet",
        repo=REPO,
        original_root=tmp_path,
        client=bucket.client(),
    )
    for name in (
        "runner.log",
        "orchestrator.log",
        "census_pre.txt",
        "devices_pre.txt",
        "census_failure_exit.txt",
        "devices_failure_exit.txt",
    ):
        (root / name).write_text("original controller failure and cleanup fixture\n")
    transport.publish_controller_failure(root, client=bucket.client())
    original_objects = dict(bucket.objects)

    # Initialize actual schema before redirecting canonical DB access.
    sys.path.insert(0, str(REPO / "bench"))
    import provenance

    db = tmp_path / "canonical-test.db"
    provenance.connect(str(db)).close()
    connect = sqlite3.connect

    def isolated_connect(database, *args, **kwargs):
        value = str(database)
        if value == "file:/home/gianl/glm-tpu/bench/results.db?mode=ro":
            database = f"file:{db}?mode=ro"
        elif value == "/home/gianl/glm-tpu/bench/results.db":
            database = str(db)
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", isolated_connect)
    opened = builtins.open
    locks = []

    def isolated_open(path, *args, **kwargs):
        if str(path) in (
            "/home/gianl/glm-run/.glm_pod_workload.lock",
            "/home/gianl/.glm-tpu-rsync.lock",
        ):
            locks.append(str(path))
            path = tmp_path / Path(path).name
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", isolated_open)
    check_output = subprocess.check_output
    original_source = (
        REPO / "scripts/greenfield/run_fp8_matmul_microbench.sh"
    ).read_text()

    def checked(command, *args, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return b""
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "b" * 40 + "\n"
        if command[:3] == ["git", "ls-remote", "origin"]:
            return "b" * 40 + "\trefs/heads/rewrite/topology-first-decode\n"
        if command[:2] == ["git", "show"]:
            assert (
                command[2] == PIN + ":scripts/greenfield/run_fp8_matmul_microbench.sh"
            )
            return original_source
        if command[:4] == ["gcloud", "storage", "buckets", "describe"]:
            return "US-CENTRAL2\n"
        return check_output(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", checked)
    run = subprocess.run
    censuses = []

    def no_launch(command, *args, **kwargs):
        if command[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["bash", "-c"]:
            label = command[2].rsplit("\nstrict_census ", 1)[1]
            assert label in ("recovery_pre", "post")
            censuses.append(label)
            for prefix in ("census", "devices"):
                (root / f"{prefix}_{label}.txt").write_text(
                    "eight clean hosts fixture\n"
                )
            return subprocess.CompletedProcess(command, 0)
        if command[0] == "gcloud":
            raise AssertionError("unexpected cloud action")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_launch)

    def forbidden(*args, **kwargs):
        raise AssertionError("fresh collector or TPU campaign called during recovery")

    monkeypatch.setattr(campaign, "collect", forbidden)
    monkeypatch.setattr(campaign, "campaign", forbidden)
    monkeypatch.setattr(sys, "argv", ["recovery", "--tag", TAG, "--pin", PIN])
    recovery.main()
    assert len(locks) == 2 and censuses == ["recovery_pre", "post"]
    with connect(db) as c:
        assert c.execute("select correct,score,latency_ms from items").fetchall() == [
            (None, None, None)
        ]
    with connect(root / "results_ckpt.db") as c:
        assert c.execute("pragma integrity_check").fetchone()[0] == "ok"
        assert c.execute("select count(*) from items").fetchone()[0] == 1
    assert (root / "hlo/candidate.optimized_hlo.txt").read_bytes() == (
        root / "fleet/rank0/dense01.optimized_hlo.txt"
    ).read_bytes()
    assert json.loads((root / "recovery.json").read_text())["model_rerun"] is False
    assert "results_ckpt.db" in (root / "evidence.sha256").read_text()
    assert all(bucket.objects[k] == v for k, v in original_objects.items())
    terminal = json.loads(bucket.objects[f"results/{TAG}/SUCCESS"][1])
    assert terminal["diagnostic_only"] and not terminal["performance_claim"]
    monkeypatch.setattr(sys, "argv", ["recovery", "--tag", TAG, "--pin", PIN])
    with pytest.raises(ValueError, match="already-accounted"):
        recovery.main()
