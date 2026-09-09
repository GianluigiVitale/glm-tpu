"""Existing campaign to bounded originals to real wrapper SQLite, CPU fixtures."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from google.cloud import storage
import pytest

from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield import ws32_dense_frontier_transport as transport
from tests.greenfield.validation.test_ws32_dense_frontier_transport import (
    Bucket,
    originals,
    TAG,
    PIN,
)

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts/greenfield/run_fp8_matmul_microbench.sh"


def test_actual_campaign_transport_wrapper_db(tmp_path, monkeypatch):
    root = tmp_path / TAG
    root.mkdir()
    bucket = Bucket()
    monkeypatch.setattr(storage, "Client", bucket.client)
    monkeypatch.setattr(campaign, "run_root", lambda tag: root)

    def forbidden(*a, **k):
        raise AssertionError("dense path entered historical layer inference")

    monkeypatch.setattr(campaign, "layer_from_tag", forbidden)
    for rank in range(8):
        originals(root / f"rank{rank}", rank)
        campaign.publish_rank(TAG, rank)
    replayed = []

    def replay(fleet, records, **kwargs):
        # Actual producer/computation replay separately covered by the evidence
        # suite; this integration substitutes only that expensive numerical seam.
        assert fleet == root / "fleet" and kwargs["repo"] == REPO
        assert (
            kwargs["original_root"]
            == Path("/home/gianl/glm-run")
            / transport.protocol.ORIGINAL_TAG
            / "first_window_collected"
        )
        assert [r["launch_rank"] for r in records] == list(range(8))
        assert all(
            (fleet / f"rank{r}/retained_preflight.json").is_file() for r in range(8)
        )
        replayed.append(True)
        return dict(
            reproduced=True,
            owners=32,
            hosts=8,
            model_calls_per_host=5,
            wk_calls_per_host=4,
            numerical_promotion=False,
            performance_claim=False,
        )

    monkeypatch.setattr(transport.evidence, "validate_fleet", replay)
    commands = []

    def ssh(command, *, output, **kwargs):
        commands.append(command)
        if output.name == "fleet_sync.log":
            text = "".join(f"PREFILL_SYNC_OK host{i}\n" for i in range(8))
        elif output.name == "retained_preflight.log":
            text = "".join(f"PREFILL_RETAINED_OK host{i}\n" for i in range(8))
        elif output.name == "coordinator.log":
            text = "127.0.0.1\n"
        else:
            assert output.name == "fleet_launch.log" and kwargs["timeout"] == 780
            assert "timeout --kill-after=30s 600s" in command
            assert "timeout --kill-after=10s 120s" in command
            assert "probe_ws32_prefill_layer.py" in command
            assert "publish-rank" in command and "trap upload EXIT" in command
            text = "fixture runtime; originals already published above\n"
        output.write_text(text)

    monkeypatch.setattr(campaign, "ssh", ssh)
    campaign.campaign(TAG, PIN)
    result = json.loads((root / "runner.json").read_text())
    assert len(commands) == 4
    assert (root / "hlo/candidate.optimized_hlo.txt").read_bytes() == (
        root / "fleet/rank0/dense01.optimized_hlo.txt"
    ).read_bytes()
    assert result["status"] == "SUCCESS" and result["admission_only"] is False
    for mode in ("diagnostic", "materialized", "observed", "prefix_mlp"):
        with pytest.raises(ValueError, match="another layer mode"):
            campaign.validate_record(result, PIN, **{mode: True})
    for field, value in (
        ("admission_only", True),
        ("latency", {"p50_ms": 1}),
        ("code_hash", "b" * 40),
        ("diagnostic_only", False),
        ("original_bytes", 1),
    ):
        bad = deepcopy(result)
        bad[field] = value
        with pytest.raises(ValueError):
            campaign.validate_record(bad, PIN)
    (root / "runner.json").write_text(json.dumps(result))
    accounting = next(
        block.split("\nPY\n", 1)[0]
        for block in WRAPPER.read_text().split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in block
    )
    db = root / "test.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(root),
            PIN,
            str(db),
            str(REPO),
            "1",
            transport.protocol.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-dense-wrapper-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "select correct, score, latency_ms from items"
        ).fetchall() == [(None, None, None)]
        assert connection.execute("select item_id from items").fetchall() == [
            ("dense01_frozen_b128_db604_reproduction_nine_calls_v1",)
        ]
    summary = json.loads((root / "summary.json").read_text())
    assert (
        summary["claim_scope"] == transport.NOTE
        and summary["performance_claim"] is False
    )
    assert len(replayed) >= 2


def test_shell_distinct_tag_no_samples_and_existing_leases():
    source = WRAPPER.read_text()
    prefix = source.split('[[ $(git -C "$WORKTREE" branch --show-current)', 1)[0]
    env = {
        **os.environ,
        "GLM_GREENFIELD_FP8_MATMUL_KERNEL": transport.protocol.KERNEL,
        "GLM_GREENFIELD_FP8_MATMUL_TAG": TAG,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            prefix
            + '\nprintf "%s %s %s %s %s" "$TAG_STEM" "$BOUNDED_PREFILL" "$WARMUP" "$ITERATIONS" "$WINDOW_NUMERICAL"',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert result.stdout == "ws32_dense_frontier_d01 1 0 0 0"
    assert "readonly RESULTS_DB=/home/gianl/glm-tpu/bench/results.db" in source
    assert "exec 9>/home/gianl/glm-run/.glm_pod_workload.lock" in source
    assert "exec 8>/home/gianl/.glm-tpu-rsync.lock" in source
    assert source.index("strict_census pre ||") < source.index("started=$(date +%s)")
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_whole_failure_archive_budget_and_no_second_full_copy(tmp_path, monkeypatch):
    root = tmp_path / TAG
    originals(root, 0)
    bucket = Bucket()
    transport.publish_controller_failure(root, client=bucket.client())
    assert all(
        name.startswith(f"results/{TAG}/") and "/diagnostic/" not in name
        for name in bucket.objects
    )
    assert not any(name.endswith("/SUCCESS") for name in bucket.objects)
    before = dict(bucket.objects)
    transport.publish_controller_failure(root, client=bucket.client())
    assert bucket.objects == before
    (root / "unsafe").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        transport.archive_inventory(root)
    (root / "unsafe").unlink()
    monkeypatch.setattr(
        transport, "MAX_ARCHIVE_BYTES", transport.MAX_FLEET_BYTES + (1 << 20)
    )
    with pytest.raises(ValueError, match="whole archive"):
        transport.publish_controller_failure(root, client=bucket.client())
    assert bucket.objects == before


def test_partial_normal_archive_then_failure_log_preserves_remaining_files(tmp_path):
    root = tmp_path / TAG
    originals(root, 0)
    log = root / "orchestrator.log"
    log.write_bytes(b"normal archive begins\n")
    bucket = Bucket()
    name = f"results/{TAG}/orchestrator.log"
    transport.publish_exact(
        bucket, name, log, transport.digest_file(log), compressed=False
    )
    original = bucket.objects[name]
    log.write_bytes(b"normal archive begins\nFAILED\n")
    transport.publish_controller_failure(root, client=bucket.client())
    assert bucket.objects[name] == original
    assert (
        bucket.objects[f"results/{TAG}/failure_orchestrator.log"][1] == log.read_bytes()
    )
    assert (
        bucket.objects[f"results/{TAG}/wide_final.npz"][1]
        == (root / "wide_final.npz").read_bytes()
    )
