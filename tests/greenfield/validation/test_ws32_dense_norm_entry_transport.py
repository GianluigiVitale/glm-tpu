"""Existing entry + actual18-call writer; exact transport with fixture cloud/math.

Independent numerical/graph replay is covered by norm_evidence/norm_fleet;
these tests exercise its transport boundary, not real TPU execution.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import run_short_decoder_ws32 as runner
from scripts.greenfield import ws32_dense_frontier_entry as entry
from scripts.greenfield import ws32_dense_frontier_transport as transport
from scripts.greenfield import ws32_dense_norm_admission as admission
from scripts.greenfield import ws32_dense_norm_evidence as evidence
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from tests.greenfield.validation.test_ws32_dense_frontier_transport import Bucket
from tests.greenfield.validation.test_ws32_dense_norm_execution import staged

TAG = "greenfield_fp8_ws32_dense_norm_d01_20260909T180000000000000Z"
PIN = "a" * 40


@pytest.mark.parametrize(
    "failure", [None, "retained", "own", "terminal_peer", "missing_call", "scope"]
)
def test_entry_actual18calls_and_failure_votes(tmp_path, monkeypatch, failure):
    run, actual, events = staged(
        tmp_path, monkeypatch, failure if failure in ("retained", "own") else None
    )
    (tmp_path / "retained_preflight.json").write_text(
        json.dumps(
            dict(
                tag=TAG,
                code_hash=PIN,
                launch_rank=0,
                hostname=entry.socket.gethostname(),
                protocol=protocol.PROTOCOL,
            )
        )
    )
    monkeypatch.setattr(
        entry.model_admission, "require_acquired_model_source", lambda *a, **k: None
    )
    runtime = object()
    monkeypatch.setattr(runner, "_initialize_runtime", lambda args: runtime)

    def execute_bound(**kwargs):
        assert kwargs["runtime"] is runtime
        assert kwargs["inspect_program"] is admission.inspect_program
        assert kwargs["record"]["protocol"] == protocol.PROTOCOL
        assert kwargs["record"]["profile"] == protocol.PROFILE
        try:
            run()
            if failure == "missing_call":
                actual["call_evidence"].pop()
            elif failure == "scope":
                actual["dense_norm"]["cause_claim"] = True
        finally:
            kwargs["record"].update(actual)

    monkeypatch.setattr(entry.runtime_module, "execute_bound", execute_bound)
    from jax.experimental import multihost_utils

    votes = []

    def vote(passed):
        votes.append(bool(passed))
        return np.asarray([False] if failure == "terminal_peer" else [passed])

    monkeypatch.setattr(multihost_utils, "process_allgather", vote)
    args = NS(expected_code_hash=PIN, process_id=0, output_dir=tmp_path)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            entry.execute(args, tag=TAG, repo=Path.cwd())
    else:
        assert entry.execute(args, tag=TAG, repo=Path.cwd()) == 0
    saved = json.loads((tmp_path / "runner.json").read_bytes())
    assert saved["status"] == ("DIAGNOSTIC_FAILED" if failure else entry.STATUS)
    if failure not in ("retained", "own"):
        assert votes == [failure not in ("missing_call", "scope")]
        assert (tmp_path / "cross_comparison.json").exists()
        assert len(list(tmp_path.glob("*.npz"))) == 23


def originals(source, rank):
    assert set(transport.NORM_CAPSULE_NAMES) == set(evidence.ALL_NAMES)
    source.mkdir()
    for name in transport.NORM_FILES:
        raw = (
            json.dumps(
                dict(
                    launch_rank=rank,
                    code_hash=PIN,
                    protocol=protocol.PROTOCOL,
                    programs={
                        n: dict(optimized_hlo_sha256="b" * 64)
                        for n in protocol.PROGRAMS
                    },
                )
            ).encode()
            if name == "runner.json"
            else name.encode()
        )
        (source / name).write_bytes(raw)


def published(root):
    bucket = Bucket()
    for rank in range(8):
        source = root / f"source{rank}"
        originals(source, rank)
        transport.publish_rank(tag=TAG, rank=rank, root=source, client=bucket.client())
    bucket.events.clear()
    return bucket


def test_norm_generation_transport_recovery_accounting(tmp_path, monkeypatch):
    bucket = published(tmp_path)
    destination = tmp_path / "fleet"
    calls = []
    references = dict(
        original_root=tmp_path / "db604", norm_original_root=tmp_path / "db605"
    )

    def replay(root, records, **kwargs):
        assert root == destination
        assert kwargs == dict(pin=PIN, tag=TAG, repo=tmp_path, **references)
        assert [r["launch_rank"] for r in records] == list(range(8))
        for rank in range(8):
            for name in transport.NORM_FILES:
                assert (root / f"rank{rank}" / name).read_bytes() == (
                    tmp_path / f"source{rank}" / name
                ).read_bytes()
        calls.append(True)
        return dict(reproduced=True, numerical_promotion=False, performance_claim=False)

    monkeypatch.setattr(transport.evidence, "validate_fleet", replay)
    kwargs = dict(tag=TAG, pin=PIN, root=destination, repo=tmp_path, **references)
    result = transport.collect(**kwargs, client=bucket.client())
    assert (
        result["kernel"] == protocol.KERNEL and result["protocol"] == protocol.PROTOCOL
    )
    assert result["latency"] is None and not result["performance_claim"]
    assert [n for op, n in bucket.events if op == "download"][:8] == [
        f"results/{TAG}/workers/rank{r}/worker_receipts.json" for r in range(8)
    ]
    transport.validate_record(
        result, PIN, root=destination, repo=tmp_path, **references
    )
    bucket.events.clear()
    assert transport.replay_collected(**kwargs, client=bucket.client()) == result
    assert len(calls) == 3
    assert [n for op, n in bucket.events if op == "download"] == [
        f"results/{TAG}/workers/rank{r}/worker_receipts.json" for r in range(8)
    ]
    assert not any(op == "upload" for op, _ in bucket.events)
    changed = deepcopy(result)
    changed["kernel"] = transport.protocol.KERNEL
    with pytest.raises(ValueError, match="aggregate"):
        transport.validate_record(
            changed, PIN, root=destination, repo=tmp_path, **references
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "model_budget", "generation"]
)
def test_norm_receipts_refuse_before_payload(tmp_path, monkeypatch, mutation):
    bucket = published(tmp_path)
    key = f"results/{TAG}/workers/rank7/worker_receipts.json"
    gen, raw = bucket.objects[key]
    values = json.loads(raw)
    if mutation == "missing":
        values.pop()
    elif mutation == "extra":
        values.append(
            {**values[-1], "name": f"results/{TAG}/workers/rank7/cross_0.npz.pending"}
        )
    elif mutation == "duplicate":
        values[-1] = values[0]
    elif mutation == "model_budget":
        for v in values:
            if v["name"].endswith("cross_0.npz"):
                v["size"] = (128 << 20) + 1
    else:
        values[-1]["generation"] = "0"
    bucket.objects[key] = (gen, json.dumps(values).encode())
    with pytest.raises(ValueError):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path / "db604",
            norm_original_root=tmp_path / "db605",
            client=bucket.client(),
        )
    assert all(
        n.endswith("worker_receipts.json")
        for op, n in bucket.events
        if op == "download"
    )
    assert not (tmp_path / "fleet").exists()


def test_norm_pending_and_required_originals(tmp_path):
    bucket = Bucket()
    root = tmp_path / TAG
    root.mkdir()
    (root / "own_wide_final.npz.pending").write_bytes(b"partial")
    rows = transport.publish_rank(tag=TAG, rank=0, root=root, client=bucket.client())
    assert len(rows) == 1 and rows[0]["name"].endswith("own_wide_final.npz.pending")
    assert transport._kind("own_wide_final.npz.pending") == "model"
    assert transport._kind("cross_96.npz") == "model"
    assert len([n for n in transport.NORM_FILES if n.endswith(".npz")]) == 23
    with pytest.raises(ValueError, match="retained-original root"):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path,
            client=bucket.client(),
        )
    assert root / "own_wide_final.npz.pending" in transport.archive_inventory(root)
