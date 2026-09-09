"""Actual five-call entry and bounded transport; explicit fixture math/cloud."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import run_short_decoder_ws32 as runner
from scripts.greenfield import ws32_dense_frontier_entry as entry
from scripts.greenfield import ws32_dense_frontier_transport as transport
from scripts.greenfield import ws32_dense_canonical as canonical
from scripts.greenfield import ws32_dense_canonical_admission as admission
from scripts.greenfield import ws32_dense_canonical_evidence as evidence
from tests.greenfield.validation.test_ws32_dense_frontier_transport import Bucket
from tests.greenfield.validation.test_ws32_dense_canonical_execution import staged

TAG = "greenfield_fp8_ws32_dense_canonical_d01_20260909T210000000000000Z"
PIN = "a" * 40


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "reproduction",
        "health",
        "terminal_peer",
        "missing_call",
        "scope",
        "source",
        "reference",
    ],
)
def test_actual_fivecall_entry_and_refusals(tmp_path, monkeypatch, failure):
    run, actual, events = staged(
        tmp_path,
        monkeypatch,
        failure if failure in ("reproduction", "health") else None,
    )
    (tmp_path / "retained_preflight.json").write_text(
        json.dumps(
            dict(
                tag=TAG,
                code_hash=PIN,
                launch_rank=0,
                hostname=entry.socket.gethostname(),
                protocol=canonical.PROTOCOL if failure != "reference" else "wrong",
            )
        )
    )

    def source(repo):
        if failure == "source":
            raise ValueError("source fixture refusal")

    monkeypatch.setattr(canonical, "require_source", source)
    monkeypatch.setattr(
        entry.model_admission,
        "require_acquired_model_source",
        lambda *a, **k: pytest.fail("historical source guard"),
    )
    initialized = []
    monkeypatch.setattr(
        runner, "_initialize_runtime", lambda args: initialized.append(True)
    )

    def execute_bound(**kwargs):
        assert kwargs["inspect_program"] is admission.inspect_program
        assert kwargs["record"]["profile"] == canonical.PROFILE
        try:
            run()
            if failure == "missing_call":
                actual["call_evidence"].pop()
            if failure == "scope":
                actual["dense_canonical"]["comparison"]["token11_cause_proven"] = True
        finally:
            kwargs["record"].update(actual)

    monkeypatch.setattr(entry.runtime_module, "execute_bound", execute_bound)
    from jax.experimental import multihost_utils

    monkeypatch.setattr(
        multihost_utils,
        "process_allgather",
        lambda passed: np.asarray([False if failure == "terminal_peer" else passed]),
    )
    args = NS(expected_code_hash=PIN, process_id=0, output_dir=tmp_path)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            entry.execute(args, tag=TAG, repo=Path.cwd())
    else:
        assert entry.execute(args, tag=TAG, repo=Path.cwd()) == 0
        journal = [
            json.loads(l)
            for l in (tmp_path / "compile_journal.jsonl").read_bytes().splitlines()
        ]
        assert [r["stage"] for r in journal] == evidence.expected_stages()
    if failure in ("source", "reference"):
        assert not initialized and not any(e[0] == "dense_dispatch" for e in events)
    else:
        assert len(list(tmp_path.glob("*.npz"))) == 5
    saved = json.loads((tmp_path / "runner.json").read_bytes())
    assert saved["status"] == ("DIAGNOSTIC_FAILED" if failure else entry.STATUS)


def originals(source, rank):
    source.mkdir()
    for name in transport.CANONICAL_FILES:
        raw = (
            json.dumps(
                dict(
                    launch_rank=rank,
                    code_hash=PIN,
                    protocol=canonical.PROTOCOL,
                    programs={
                        n: dict(optimized_hlo_sha256="b" * 64)
                        for n in canonical.PROGRAMS
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


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "model_budget", "generation", "reference"]
)
def test_receipt_refusal_before_any_payload(tmp_path, mutation):
    bucket = published(tmp_path)
    key = f"results/{TAG}/workers/rank7/worker_receipts.json"
    generation, raw = bucket.objects[key]
    values = json.loads(raw)
    if mutation == "missing":
        values.pop()
    elif mutation == "extra":
        values.append(
            {
                **values[-1],
                "name": f"results/{TAG}/workers/rank7/{canonical.CAPSULE}.npz.pending",
            }
        )
    elif mutation == "model_budget":
        next(v for v in values if v["name"].endswith(canonical.CAPSULE + ".npz"))[
            "size"
        ] = (128 << 20) + 1
    elif mutation == "generation":
        values[-1]["generation"] = "0"
    bucket.objects[key] = generation, json.dumps(values).encode()
    with pytest.raises(ValueError):
        transport.collect(
            tag=TAG,
            pin=PIN,
            root=tmp_path / "fleet",
            repo=tmp_path,
            original_root=tmp_path / "db604",
            client=bucket.client(),
            **(
                {}
                if mutation == "reference"
                else dict(norm_original_root=tmp_path / "db605")
            ),
        )
    assert all(
        n.endswith("worker_receipts.json")
        for op, n in bucket.events
        if op == "download"
    )
    assert not (tmp_path / "fleet").exists()


def test_pending_capsule_is_model_not_auxiliary(tmp_path):
    bucket = Bucket()
    name = canonical.CAPSULE + ".npz.pending"
    (tmp_path / name).write_bytes(b"partial original")
    rows = transport.publish_rank(
        tag=TAG, rank=0, root=tmp_path, client=bucket.client()
    )
    assert len(rows) == 1 and rows[0]["name"].endswith(name)
    assert (
        transport._kind(name) == transport._kind(canonical.CAPSULE + ".npz") == "model"
    )
    assert len(transport.CANONICAL_FILES) == 17
    assert len([n for n in transport.CANONICAL_FILES if n.endswith(".npz")]) == 5
