"""Actual worker/collector/DB composition, fixture math and GCS only (no TPU)."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import probe_ws32_prefill_budget as entry
from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_overhead as overhead
from scripts.greenfield import prefill_window_worker as shared
from scripts.greenfield import prefill_window_acquisition as phases

PIN = "a" * 40
TAG = f"greenfield_fp8_{entry.KERNEL}_fixture"


@pytest.mark.parametrize(
    "mutation", ["rank", "pin", "tag", "path", "bounds", "existing"]
)
def test_actual_request_refuses_before_runtime(tmp_path, monkeypatch, mutation):
    import scripts.greenfield.run_short_decoder_ws32 as runtime

    root = tmp_path / TAG
    monkeypatch.setattr(
        entry,
        "run_root",
        lambda tag: root if tag == TAG else (_ for _ in ()).throw(ValueError("tag")),
    )
    monkeypatch.setattr(entry, "_git_head", lambda: PIN)
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", TAG)
    initialized = []
    monkeypatch.setattr(
        runtime, "_initialize_runtime", lambda args: initialized.append(True)
    )
    args = [
        "--expected-code-hash",
        PIN,
        "--coordinator-address",
        "127.0.0.1:8476",
        "--process-id",
        "4",
        "--output-dir",
        str(root / "rank4"),
    ]
    if mutation == "rank":
        args[5] = "8"
    elif mutation == "pin":
        args[1] = "b" * 40
    elif mutation == "tag":
        monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", "unscoped")
    elif mutation == "path":
        args[-1] = str(tmp_path)
    elif mutation == "bounds":
        monkeypatch.setenv("TPU_VISIBLE_DEVICES", "0,1,2,3")
    else:
        (root / "rank4").mkdir(parents=True)
        (root / "rank4/runner.json").write_text("existing")
    with pytest.raises((ValueError, FileExistsError)):
        entry.main(args)
    assert initialized == []


@pytest.mark.parametrize(
    "failure", ["close", "publication", "peer_finalize", "peer_terminal", "body"]
)
def test_outer_finalization_votes_before_terminal(tmp_path, monkeypatch, failure):
    record = dict(
        protocol=probe.PROTOCOL,
        profile=worker.PROFILE,
        compile_only=False,
        code_hash=PIN,
        launch_rank=0,
        jax_process_index=0,
        status="RUNNING",
    )
    original_close, original_publish = worker.BudgetJournal.close, phases._atomic_json
    votes = []

    def body(calls, mesh):
        shared.save_arrays(tmp_path / "original.npz", dict(value=np.arange(4)))
        if failure == "body":
            raise ValueError("body failure")

    def close(journal):
        original_close(journal)
        if failure == "close":
            raise OSError("close failure")

    def publish(path, value):
        if (
            failure == "publication"
            and value.get("acquisition_phases", {})
            .get("budget_finalize", {})
            .get("status")
            == "RUNNING"
        ):
            raise OSError("publication failure")
        original_publish(path, value)

    def consensus(ok):
        votes.append(ok)
        latest = list(record.get("acquisition_phases", {}))[-1]
        return (
            ok
            and not (failure == "peer_finalize" and latest == "budget_finalize")
            and not (failure == "peer_terminal" and latest == "budget_terminal")
        )

    monkeypatch.setattr(worker, "run_budget_campaign", body)
    monkeypatch.setattr(worker.BudgetJournal, "close", close)
    monkeypatch.setattr(phases, "_atomic_json", publish)
    fake_jax = SimpleNamespace(
        local_devices=lambda: [SimpleNamespace(id=i) for i in range(4)]
    )
    with pytest.raises(
        (ValueError, OSError, RuntimeError), match="failure|peer failed"
    ):
        entry.execute_budget(
            tmp_path,
            record,
            jax=fake_jax,
            mesh=None,
            physical_mesh=SimpleNamespace(flattened_device_ids=range(32)),
            consensus=consensus,
        )
    assert (tmp_path / "original.npz").is_file()
    assert votes
    if failure != "peer_terminal":
        assert "budget_terminal" not in record["acquisition_phases"]


def actual_worker_fixture(tmp_path, monkeypatch):
    """Run actual CLI→runtime envelope→overhead→44calls→finalization once."""
    import jax
    from jax.experimental import multihost_utils
    from scripts.greenfield import probe_ws32_prefill_layer as layer
    from scripts.greenfield import run_short_decoder_ws32 as runtime
    from scripts.greenfield import ws32_batched_prefill_runner as runner
    from tests.greenfield.hlo.test_prefill_budget_worker import make_worker
    from tests.greenfield.hlo.test_prefill_budget_overhead import Array, state

    seed = tmp_path / "seed"
    seed.mkdir()
    old_calls, compiler, events = make_worker(seed, monkeypatch)
    old_calls.journal.close()
    root = tmp_path / TAG
    physical, captures = campaign.topology_bindings()
    # Actual captured launch4 owns devices0..3, and is JAX process0.
    monkeypatch.setattr(
        jax, "local_devices", lambda: [SimpleNamespace(id=i) for i in range(4)]
    )
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(entry, "run_root", lambda tag: root)
    monkeypatch.setattr(entry, "_git_head", lambda: PIN)
    monkeypatch.setattr(entry.socket, "gethostname", lambda: captures[4]["hostname"])
    monkeypatch.setattr(
        entry, "version", lambda name: {"jax": "0.10.1", "libtpu": "0.0.41"}[name]
    )
    monkeypatch.setattr(
        runtime,
        "_initialize_runtime",
        lambda args: (
            jax,
            "mesh",
            physical,
            SimpleNamespace(topology_hash=entry.TOPOLOGY_SHA),
            entry.FLEET_SHA,
        ),
    )
    monkeypatch.setattr(
        multihost_utils, "process_allgather", lambda value: np.repeat(value, 8)
    )
    monkeypatch.setattr(layer, "compile_program", compiler)
    monkeypatch.setattr(
        probe,
        "measure_initial_state",
        lambda mesh, config, prompt_length, clock: (
            state(config, prompt_length),
            dict(
                capacity=config.context_capacity,
                prompt_length=prompt_length,
                cache_initialization_seconds=0.0,
                weights_loaded=False,
                initialized_prefix_length=0,
                model_ttft_measured=False,
            ),
        ),
    )
    monkeypatch.setattr(
        overhead,
        "capture_identified_device_memory",
        lambda devices: [
            dict(
                device_id=i,
                process_index=0,
                platform="tpu",
                bytes_in_use=4_000_000_000,
                peak_bytes_in_use=4_000_000_000,
                bytes_limit=33_014_398_976,
            )
            for i in range(4)
        ],
    )
    monkeypatch.setattr(
        runner, "replicated", lambda mesh, value: Array(np.asarray(value))
    )
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", TAG)
    assert (
        entry.main(
            [
                "--expected-code-hash",
                PIN,
                "--coordinator-address",
                "127.0.0.1:8476",
                "--process-id",
                "4",
                "--output-dir",
                str(root / "rank4"),
            ]
        )
        == 0
    )
    record = json.loads((root / "rank4/runner.json").read_text())
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "execute"]) == 44
    assert (
        record["status"] == "SUCCESS"
        and record["current_phase"] == "budget/dsa_complete"
    )
    return root, record, physical, captures


def clone_synthetic_owner(source, dest, record, *, rank, capture, physical):
    """Fresh analytic arrays for synthetic peer records, not real fleet evidence."""
    dest.mkdir()
    mapping = dict(zip(range(4), capture["local_device_ids"], strict=True))
    process = capture["jax_process_index"]

    def remap(value):
        if isinstance(value, list):
            return [remap(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: remap(v) for k, v in value.items()}
        if "device_id" in result:
            result["device_id"] = mapping[result["device_id"]]
        if "process_index" in result:
            result["process_index"] = process
        for key in ("owners", "local_payload_bytes"):
            if key in value:
                result[key] = {
                    str(mapping[int(k)]): remap(v) for k, v in value[key].items()
                }
        return result

    result = remap(record)
    result.update(
        launch_rank=rank, jax_process_index=process, hostname=capture["hostname"]
    )
    slots = {d: s for s, d in enumerate(physical.flattened_device_ids)}
    result["local_device_slots"] = [
        dict(device_id=d, device_slot=slots[d]) for d in mapping.values()
    ]
    if process != 0:
        result["budget_overhead"]["delivery"] = []
    for call in result["call_evidence"]:
        call["budget"] = worker.memory_budget(
            call["census"], call["compiled_memory"], active_graph=call["graph"]
        )
    journal = [
        json.loads(line)
        for line in (source / "compile_journal.jsonl").read_text().splitlines()
    ]
    journal[0]["identity"]["launch_rank"] = rank
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in journal).encode()
    (dest / "compile_journal.jsonl").write_bytes(raw)
    result["compile_journal_sha256"] = sha256(raw).hexdigest()
    for name in worker.PROGRAMS:
        for suffix in ("stablehlo.mlir", "optimized_hlo.txt"):
            os.link(source / f"{name}.{suffix}", dest / f"{name}.{suffix}")
    for name in result["budget_originals"]:
        with np.load(source / name, allow_pickle=False) as original:
            arrays = {
                f"device{new}_{field}": original[f"device{old}_{field}"]
                for old, new in mapping.items()
                for field in ("positions", "valid_counts", "scores", "health")
            }
        result["budget_originals"][name] = shared.save_arrays(dest / name, arrays)
    return result


def test_actual_cli_publication_fleet_replay_and_database(
    tmp_path, monkeypatch, capsys
):
    root, template, physical, captures = actual_worker_fixture(tmp_path, monkeypatch)
    actual_log = capsys.readouterr().out
    assert "PREFILL_BUDGET rank=4" in actual_log and "complete" in actual_log
    assert "GREENFIELD_WS32_ACQUISITION" in actual_log
    for rank, capture in enumerate(captures):
        dest = root / f"rank{rank}"
        r = (
            template
            if rank == 4
            else clone_synthetic_owner(
                root / "rank4",
                dest,
                template,
                rank=rank,
                capture=capture,
                physical=physical,
            )
        )
        (dest / "runner.json").write_text(json.dumps(r))
        (dest / "worker.log").write_text(
            actual_log if rank == 4 else "Synthetic CPU fixture, NOT TPU evidence\n"
        )
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication

    blobs = {}
    downloads = []

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, data):
            self.name, self.data, self.size = name, data, len(data)

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            downloads.append(self.name)
            return self.data

    bucket = SimpleNamespace(
        get_blob=lambda name: blobs.get(name), blob=lambda name, generation: blobs[name]
    )

    def named_bucket(name):
        assert name == "driftbench-dsv4-uc"
        return bucket

    def publish(bucket, name, path, digest, *, compressed):
        assert not compressed
        b = Blob(name, path.read_bytes())
        blobs[name] = b
        return dict(
            name=name,
            generation=str(b.generation),
            size=b.size,
            crc32c=b.crc32c,
            original_sha256=sha256(b.data).hexdigest(),
        )

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=named_bucket))
    monkeypatch.setattr(publication, "publish_exact", publish)
    monkeypatch.setattr(campaign, "run_root", lambda tag: root)
    for rank in range(8):
        campaign.publish_rank(TAG, rank)
    result = campaign.collect(TAG, PIN)
    assert all(name.endswith("worker_receipts.json") for name in downloads[:8])
    campaign.validate_record(result, PIN, root)
    assert len(result["dsa_wall"]["cases"]) == 6
    assert len(result["overhead_wall"]["delivery"]) == 5
    assert len(result["overhead_wall"]["initialization"]) == 2
    for change in ("owner", "rank", "finalize", "wall", "overhead"):
        bad = deepcopy(result)
        if change == "owner":
            bad["workers"][0]["local_device_slots"][0]["device_slot"] = 31
        elif change == "rank":
            bad["workers"][0]["jax_process_index"] = 0
        elif change == "finalize":
            del bad["workers"][0]["acquisition_phases"]["budget_finalize"]
        elif change == "wall":
            bad["dsa_wall"]["cases"][probe.cases()[0].name]["p50_seconds"] += 1
        else:
            del bad["workers"][0]["budget_overhead"]
        with pytest.raises(ValueError):
            campaign.validate_record(bad, PIN, root)
    (root / "runner.json").write_text(json.dumps(result))
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    db = tmp_path / "budget.db"
    monkeypatch.setattr(
        sys,
        "argv",
        ["accounting", str(root), PIN, str(db), str(Path.cwd()), "1", entry.KERNEL],
    )
    exec(
        compile(accounting, "<actual-budget-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            (
                "dsa_long_prefix_six_cases_fresh_cache_input_local_ack_v1",
                None,
                None,
                None,
            )
        ]
    assert (
        json.loads((root / "summary.json").read_text())["claim_scope"] == campaign.NOTE
    )


def test_fixed_launch_watchdogs_and_distributed_path():
    command = campaign.launch_command(TAG, PIN, "10.0.0.1:8476")
    assert "timeout --kill-after=30s 900s" in command
    assert "timeout --kill-after=10s 120s" in command
    assert "probe_ws32_prefill_budget.py" in command
    assert "TPU_VISIBLE_DEVICES" not in command
    assert "checkpoint" not in command and "load_ws32" not in command
    assert campaign.SSH_SECONDS >= campaign.WORKER_SECONDS + 30 + 120 + 10
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert "[[ $BUDGET_BASELINE == 0 ]] || BOUNDED_PREFILL=1" in wrapper
    assert wrapper.index("ws32_prefill_budget_campaign campaign") < wrapper.index(
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1"
    )
