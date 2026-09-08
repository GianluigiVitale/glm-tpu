"""Actual acquisition compiler/journal/controller/accounting boundaries, CPU only."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import jax
import pytest

from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import probe_ws32_prefill_layer as worker
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.probe_ws32_prefill_moe import FLEET_SHA, MESH_SHA, TOPOLOGY_SHA


class Program:
    """Compilation fixture: attempting ANY executable call fails this test."""

    def __init__(self, name):
        self.name = name

    def lower(self, *values):
        name = self.name

        class Compiled:
            def as_text(self):
                return f"HloModule {name}\nENTRY %main {{\n%p = f32[2] parameter(0)\nROOT %out = f32[2] copy(%p)\n}}"

            def memory_analysis(self):
                return SimpleNamespace(
                    **dict(zip(acquisition.MEMORY_KEYS, (1000, 100, 0, 200, 300)))
                )

            def __call__(self, *_):
                raise AssertionError("acquisition executed a model or WK program")

        return SimpleNamespace(
            compiler_ir=lambda **_: f"module @{name} {{}}", compile=Compiled
        )


def compile_fixture(root, rank, monkeypatch, *, inspector=acquisition.inspect_graph):
    root.mkdir()
    devices = [SimpleNamespace(id=i) for i in range(4 * rank, 4 * rank + 4)]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(
        worker,
        "_memory_stats",
        lambda _: dict(bytes_in_use=1000, peak_bytes_in_use=2000, bytes_limit=30000),
    )
    record = dict(
        protocol=acquisition.PROTOCOL, code_hash="b" * 40, launch_rank=rank, programs={}
    )
    programs = tuple((name, Program(name), ()) for name in acquisition.PROGRAMS)
    compiled = acquisition.acquire_programs(
        programs,
        root=root,
        record=record,
        consensus=lambda passed: passed,
        compiler=worker.compile_program,
        inspector=inspector,
    )
    assert len(compiled) == 4
    record["compile_journal_sha256"] = sha256(
        (root / "compile_journal.jsonl").read_bytes()
    ).hexdigest()
    return record


def test_acquisition_compiles_all_graphs_without_any_execution(tmp_path, monkeypatch):
    record = compile_fixture(tmp_path / "rank0", 0, monkeypatch)
    acquisition.validate_files(tmp_path / "rank0", json.loads(json.dumps(record)))
    assert tuple(record["programs"]) == acquisition.PROGRAMS
    assert all(
        p["inventory"]["numerical_execution_authorized"] is False
        for p in record["programs"].values()
    )


def test_first_layer_parser_refusal_keeps_both_graphs_and_original_memory(
    tmp_path, monkeypatch
):
    real = acquisition.inspect_graph

    def refuse(hlo):
        if "HloModule candidate" in hlo:
            raise ValueError("unregistered candidate helper")
        return real(hlo)

    monkeypatch.setattr(acquisition, "inspect_graph", refuse)
    root = tmp_path / "rank0"
    record = compile_fixture(root, 0, monkeypatch, inspector=refuse)
    assert (
        record["programs"]["candidate"]["inspection_error"]
        == "ValueError: unregistered candidate helper"
    )
    assert "inventory" in record["programs"]["control"]
    assert (root / "candidate.optimized_hlo.txt").is_file()
    assert (root / "control.optimized_hlo.txt").is_file()
    acquisition.validate_files(root, record)
    stages = [
        json.loads(line)
        for line in (root / "compile_journal.jsonl").read_text().splitlines()
    ]
    assert len([r for r in stages if r["stage"] == "compiled"]) == 4
    assert len([r for r in stages if r["stage"] == "inspection_failed"]) == 1


@pytest.mark.parametrize("local_failure", [False, True])
def test_load_or_peer_failure_votes_and_stops_before_next_phase(
    tmp_path, local_failure
):
    votes, calls = [], []

    def action():
        calls.append("load")
        if local_failure:
            raise ValueError("bad selected leaf")

    def consensus(ok):
        votes.append(ok)
        return False

    with pytest.raises((ValueError, RuntimeError)):
        acquisition.fleet_step(
            "load", action, record={}, root=tmp_path, consensus=consensus
        )
        calls.append("compile")
    assert calls == ["load"] and votes == [not local_failure]
    assert json.loads((tmp_path / "runner.json").read_text())["acquisition_phases"][
        "load"
    ]["status"] == ("FAILED" if local_failure else "COMPLETE")


def test_compile_failure_does_not_attempt_next_graph_and_keeps_partial_journal(
    tmp_path, monkeypatch
):
    names = []

    def compiler(fn, values, name, root, record, *, journal):
        names.append(name)
        if name == "candidate":
            journal.begin(name)
            raise RuntimeError("compiler failed")
        return worker.compile_program(fn, values, name, root, record, journal=journal)

    monkeypatch.setattr(worker, "_memory_stats", lambda _: {})
    record = dict(
        protocol=acquisition.PROTOCOL, code_hash="b" * 40, launch_rank=0, programs={}
    )
    with pytest.raises(RuntimeError, match="compiler failed"):
        acquisition.acquire_programs(
            tuple((name, Program(name), ()) for name in acquisition.PROGRAMS),
            root=tmp_path,
            record=record,
            consensus=lambda ok: ok,
            compiler=compiler,
        )
    assert names == ["wk_decode", "wk_promote", "candidate"]
    assert "compiled_memory" in record["programs"]["wk_promote"]
    assert (tmp_path / "compile_journal.jsonl").is_file()


def test_new_tag_and_evidence_inventory_do_not_alias_old_admission():
    tag = "greenfield_fp8_ws32_prefill_layer_window_acquisition_l6_test"
    assert worker.layer_from_tag(tag) == 6
    assert worker.pins_for_layer(6) == acquisition.PINS
    assert worker.pins_for_layer(0) == worker.pins_for_layer(3) == worker.PINS
    assert campaign.program_names(6) == acquisition.PROGRAMS
    files = campaign.evidence_files(6)
    assert len(files) == 12 and "compile_journal.jsonl" in files
    assert not any(f.endswith("npz") for f in files)
    for bad in (
        tag.replace("_l6_", "_l3_"),
        tag.replace("_acquisition_", "_baseline_"),
    ):
        with pytest.raises(ValueError):
            worker.layer_from_tag(bad)
    with pytest.raises(ValueError):
        campaign.program_names(6, diagnostic=True)


@pytest.mark.parametrize("kernel", [acquisition.KERNEL, acquisition.window.KERNEL])
def test_actual_shell_defaults_route_window_to_bounded_zero_iteration_mode(kernel):
    source = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    # Stop before cloud, leases, directory writes or even the dirty-worktree
    # check; exercise the real variable/branch expansion, not a copied formula.
    prefix = source.split('[[ $(git -C "$WORKTREE" branch --show-current)', 1)[0]
    env = dict(
        os.environ,
        GLM_GREENFIELD_FP8_MATMUL_KERNEL=kernel,
        GLM_GREENFIELD_PREFILL_LAYER="6",
        GLM_GREENFIELD_FP8_MATMUL_TAG=f"greenfield_fp8_{kernel}_l6_test",
    )
    code = (
        prefix
        + '\necho "$WINDOW_ACQUISITION $GROUPED_ADMISSION $BOUNDED_PREFILL $LAYER $WARMUP $ITERATIONS"\n'
    )
    result = subprocess.run(
        ["bash", "-c", code], env=env, text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{int(kernel == acquisition.KERNEL)} 1 1 6 0 0"
    result = subprocess.run(
        ["bash", "-c", code],
        env=dict(env, GLM_GREENFIELD_PREFILL_LAYER="3"),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2


def full_worker(record, rank, pins, ledger):
    record.update(
        status="SUCCESS",
        layer=6,
        selected_layer_ids=[6],
        rows=128,
        control_rows=32,
        context_capacity=4096,
        key_tile=512,
        iterations=0,
        latency=None,
        reference_scope=acquisition.REFERENCE_SCOPE,
        state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
        integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
        checkpoint_pins=pins,
        payload_bytes_per_chip=326079840,
        mesh_sha256=MESH_SHA,
        topology_sha256=TOPOLOGY_SHA,
        topology_fleet_sha256=FLEET_SHA,
        versions={"jax": "0.10.1", "libtpu": "0.0.41"},
        model_executable_calls=0,
        wk_executable_calls=0,
        resident_programs_at_snapshot=list(acquisition.PROGRAMS),
        memory_scope="SELECTED_WEIGHTS_AND_FOUR_COMPILED_PROGRAMS_NO_NUMERICAL_SCRATCH_OR_OUTPUTS",
        cases={},
        compile_only=True,
        numerical_execution_authorized=False,
        admission_only=False,
        diagnostic_only=True,
        performance_claim=False,
        pid=10 + rank,
        start_ticks=20 + rank,
        boot_id=f"boot{rank}",
        hostname=f"host{rank}",
        jax_process_index=rank,
        physical_device_ids=[list(range(i * 4, i * 4 + 4)) for i in range(8)],
        local_device_slots=[
            dict(
                device_id=i,
                device_slot=i,
                observed_selected_tensor_sha256=ledger[i]["selected"],
                expected_full_file_sha256_not_verified=ledger[i]["full_sha256"],
                selected_payload_bytes=326079840,
            )
            for i in range(rank * 4, rank * 4 + 4)
        ],
        device_memory_stats_including_reference=[
            dict(
                device_id=i,
                stats=dict(
                    bytes_in_use=1000, peak_bytes_in_use=2000, bytes_limit=30000
                ),
            )
            for i in range(rank * 4, rank * 4 + 4)
        ],
        hlo=dict(
            sha256=record["programs"]["candidate"]["optimized_hlo_sha256"],
            contract=dict(
                passed=True, scope="COMPILER_EVIDENCE_PRESENT_NOT_EXECUTION_ADMISSION"
            ),
        ),
    )
    return record


def test_actual_journal_fleet_json_and_wrapper_accounting(tmp_path, monkeypatch):
    pins = {"fixture_only": True}
    ledger = {
        i: dict(selected={"tensor": str(i)}, full_sha256="d" * 64) for i in range(32)
    }
    monkeypatch.setattr(campaign, "checkpoint_ledger", lambda layer: (pins, ledger))
    records = []
    for rank in range(8):
        root = tmp_path / f"rank{rank}"
        record = full_worker(
            compile_fixture(root, rank, monkeypatch), rank, pins, ledger
        )
        preflight = dict(
            code_hash="b" * 40,
            layer=6,
            launch_rank=rank,
            hostname=f"host{rank}",
            headers=[dict(device_slot=i) for i in range(rank * 4, rank * 4 + 4)],
        )
        (root / "retained_preflight.json").write_text(json.dumps(preflight))
        record["retained_preflight_sha256"] = sha256(
            (root / "retained_preflight.json").read_bytes()
        ).hexdigest()
        parsed = json.loads(json.dumps(record))
        campaign.validate_files(root, parsed)
        (root / "runner.json").write_text(json.dumps(parsed))
        (root / "worker.log").write_text("CPU fixture, no TPU execution\n")
        records.append(parsed)
    campaign.validate_workers(records, "b" * 40, layer=6, pins=pins, ledger=ledger)
    # Drive the actual generation-bound collector against an in-memory bucket.
    from google.cloud import storage

    tag = "greenfield_fp8_ws32_prefill_layer_window_acquisition_l6_fixture"
    blobs = {}

    class Blob:
        generation, crc32c = 19, "fixture-crc"

        def __init__(self, name, data):
            self.name, self.data, self.size = name, data, len(data)

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return self.data

    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        receipts = []
        for filename in campaign.evidence_files(6):
            data = (tmp_path / f"rank{rank}" / filename).read_bytes()
            blob = Blob(prefix + filename, data)
            blobs[blob.name] = blob
            receipts.append(
                dict(
                    name=blob.name,
                    generation=str(blob.generation),
                    size=blob.size,
                    crc32c=blob.crc32c,
                    original_sha256=sha256(data).hexdigest(),
                )
            )
        name = prefix + "worker_receipts.json"
        blobs[name] = Blob(name, json.dumps(receipts).encode())
    bucket = SimpleNamespace(
        get_blob=lambda name: blobs[name], blob=lambda name, generation: blobs[name]
    )

    def named_bucket(name):
        assert name == "driftbench-dsv4-uc"
        return bucket

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=named_bucket))
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path / "collected")
    result = campaign.collect(tag, "b" * 40)
    campaign.validate_record(result, "b" * 40)
    for mutate in (
        lambda r: r.update(admission_only=True),
        lambda r: r.update(numerical_execution_authorized=True),
        lambda r: r["workers"][0].update(context_capacity=1024),
        lambda r: r["workers"][0].update(wk_executable_calls=1),
        lambda r: r["workers"][0]["local_device_slots"][0].update(device_slot=4),
        lambda r: r["workers"][0]["programs"]["candidate"].update(
            optimized_hlo_sha256="a" * 64
        ),
    ):
        bad = deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            campaign.validate_record(bad, "b" * 40)
    # Real shell-embedded consumer, isolated DB: compile evidence cannot become
    # correctness=1, score=1 or a numerical/performance result.
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        block.split("\nPY\n", 1)[0]
        for block in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in block
    )
    (tmp_path / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "test.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(tmp_path),
            "b" * 40,
            str(db),
            str(Path.cwd()),
            "1",
            acquisition.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-wrapper-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "no model or WK execution" in summary["claim_scope"]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "select correct, score, latency_ms from items"
        ).fetchall() == [(None, None, None)]


@pytest.mark.parametrize("mutation", ["graph", "memory", "journal", "inventory"])
def test_original_compiler_evidence_mutations_refused(tmp_path, monkeypatch, mutation):
    root = tmp_path / "rank0"
    record = compile_fixture(root, 0, monkeypatch)
    if mutation == "graph":
        (root / "control.optimized_hlo.txt").write_text("wrong")
    elif mutation == "memory":
        record["programs"]["candidate"]["compiled_memory"]["temp_size_in_bytes"] += 1
    elif mutation == "journal":
        (root / "compile_journal.jsonl").write_text("{}\n")
    else:
        record["programs"]["candidate"]["inventory"][
            "numerical_execution_authorized"
        ] = True
    with pytest.raises(ValueError):
        acquisition.validate_files(root, record)
