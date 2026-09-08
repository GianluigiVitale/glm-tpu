"""One actual CPU continuation + synthetic eight-owner publication/DB fixture.

Do not run eight identical 59-call CPU generators: subsequent rank capsules use
the same completed journal structure with explicit fixture identities and fresh
slot-correct synthetic arrays. Actual controller, replay and accounting run for
all eight capsules. This is not a distributed model or TPU measurement.
"""

from copy import deepcopy
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_completed_window_protocol as protocol
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    encode_arrays,
)
from scripts.greenfield.probe_ws32_prefill_moe import FLEET_SHA, MESH_SHA, TOPOLOGY_SHA
from tests.greenfield.hlo.test_prefill_completed_window_worker import (
    run_cases,
    continue_from_acquisition,
)
from tests.greenfield.hlo.test_prefill_window_worker import fixture_output


@pytest.mark.parametrize(
    "stage",
    [
        "numerical_finalize",
        "close_compile_journal",
        "numerical_snapshot",
        "journal_close",
        "peer_snapshot",
    ],
)
def test_completed_finalization_refusal_preserves_originals(
    tmp_path, monkeypatch, stage
):
    from scripts.greenfield import prefill_completed_window_worker as completed

    # The full59-call lifecycle is exercised once below. This injection targets
    # only its finalization boundary, with a tiny already-completed case fixture.
    def tiny_case(calls, **kwargs):
        worker.save_arrays(
            tmp_path / "completed_case_fixture.npz", {"original": np.arange(8)}
        )

    monkeypatch.setattr(completed, "execute_cases", tiny_case)
    observed = {"fired": False}
    original_publish, original_close = (
        acquisition._atomic_json,
        completed.CompletedJournal.close,
    )

    def publish(path, value):
        if (
            not observed["fired"]
            and value.get("acquisition_phases", {}).get(stage, {}).get("status")
            == "RUNNING"
        ):
            observed["fired"] = True
            raise OSError("injected final publication failure")
        return original_publish(path, value)

    def close(journal):
        original_close(journal)
        if not observed["fired"]:
            observed["fired"] = True
            raise OSError("injected journal close failure")

    monkeypatch.setattr(acquisition, "_atomic_json", publish)
    if stage == "journal_close":
        monkeypatch.setattr(completed.CompletedJournal, "close", close)
    with run_cases(tmp_path, create_journal=False) as (calls, sequence):
        votes = []

        def consensus(ok):
            votes.append(ok)
            if stage == "peer_snapshot" and "numerical_snapshot" in calls.record.get(
                "acquisition_phases", {}
            ):
                observed["fired"] = True
                return False
            return ok

        calls.consensus = consensus
        with pytest.raises((OSError, RuntimeError), match="injected|peer failed"):
            continue_from_acquisition(tmp_path, calls, sequence)
        assert observed["fired"] and (stage == "peer_snapshot" or False in votes)
        assert sequence == [("wk_decode", -1), ("wk_promote", -1)]
        assert (tmp_path / "completed_case_fixture.npz").is_file()
        assert "terminal" not in calls.record["acquisition_phases"]


def clone_fixture(source, destination, record, *, rank, slots, process):
    """Build explicitly synthetic remote-rank originals without replaying dispatch."""
    destination.mkdir()
    original_slots = [s["device_id"] for s in record["local_device_slots"]]
    record = deepcopy(record)
    record.update(launch_rank=rank, jax_process_index=process)
    record["local_device_slots"] = [
        dict(device_id=d, device_slot=s) for d, s in slots.items()
    ]
    journal = [
        json.loads(line)
        for line in (source / "compile_journal.jsonl").read_text().splitlines()
    ]
    journal[0]["identity"]["launch_rank"] = rank
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in journal).encode()
    (destination / "compile_journal.jsonl").write_bytes(raw)
    record["compile_journal_sha256"] = sha256(raw).hexdigest()
    for name in record["programs"]:
        for form in ("stablehlo.mlir", "optimized_hlo.txt"):
            os.link(source / f"{name}.{form}", destination / f"{name}.{form}")
    for name in ("wk_decode", "wk_promote", "wk_boundary"):
        with np.load(source / f"{name}.npz", allow_pickle=False) as saved:
            arrays = {}
            for old, new in zip(original_slots, slots, strict=True):
                if name == "wk_boundary":
                    for kind in ("bf16", "fp32"):
                        arrays[f"{kind}_{new}"] = saved[f"{kind}_{old}"]
                else:
                    arrays[str(new)] = saved[str(old)]
        digest = worker.save_arrays(destination / f"{name}.npz", arrays)
        if name == "wk_boundary":
            record["wk_boundary_sha256"] = digest
        else:
            record["wk_originals"][name] = digest
    for call in record["call_evidence"]:
        for rows in (call["census"]["devices"], call["post_memory"]):
            for row, device in zip(rows, slots, strict=True):
                row.update(device_id=device, process_index=process)
        call["budget"] = assembly.memory_budget(
            call["census"], call["compiled_memory"], active_graph=call["graph"]
        )
    for case in protocol.CASES:
        with np.load(source / f"{case}.npz", allow_pickle=False) as saved:
            host = decode_arrays(saved, "input", INPUT_FIELDS)
        arrays = encode_arrays("input", host)
        for device, slot in slots.items():
            whole = fixture_output(host, slot)
            for tile in range(4):
                prefix = {
                    n: (
                        np.zeros((32, 1536), window.BF16)
                        if n == "mlp_input"
                        else (
                            whole[n]
                            if n in ("kv", "index", "repair")
                            else whole[n][tile * 32 : (tile + 1) * 32]
                        )
                    )
                    for n in protocol.PREFIX_FIELDS
                }
                arrays.update(protocol.encode(f"prefix{tile}_{device}", prefix))
                arrays.update(
                    protocol.encode(
                        f"narrow{tile}_{device}",
                        {
                            n: whole[n][tile * 32 : (tile + 1) * 32]
                            for n in protocol.SUFFIX_FIELDS
                        },
                    )
                )
            arrays.update(
                protocol.encode(
                    f"wide_{device}", {n: whole[n] for n in protocol.SUFFIX_FIELDS}
                )
            )
            for kind in ("actual", "control"):
                arrays.update(encode_arrays(f"{kind}_{device}", whole))
        path = destination / f"{case}.npz"
        digest = worker.save_arrays(path, arrays)
        replay = protocol.replay_case(path, case=case, slots_by_device=slots)
        record["cases"][case] = dict(
            npz_sha256=digest, replay=replay, passed=True, complete=True
        )
    return record


def test_completed_numerical_mode_is_distinct():
    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_fixture"
    assert acquisition.is_completed_numerical_tag(tag)
    assert acquisition.is_window_tag(tag) and campaign.layer_from_tag(tag) == 6
    assert not acquisition.is_numerical_tag(tag) and not acquisition.is_acquisition_tag(
        tag
    )
    assert not acquisition.is_completed_numerical_tag(tag.replace("_l6_", "_l3_"))
    assert len(campaign.evidence_files(6, completed_numerical=True)) == 28
    for key in (
        "diagnostic",
        "materialized",
        "prefix_mlp",
        "observed",
        "window_numerical",
        "boundary_diagnostic",
        "completed_window",
    ):
        with pytest.raises(ValueError):
            campaign.evidence_files(6, completed_numerical=True, **{key: True})
    with pytest.raises(ValueError):
        campaign.evidence_files(3, completed_numerical=True)


def test_actual_probe_selects_completed_scope_before_jax_and_forwards_flags(
    tmp_path, monkeypatch
):
    from scripts.greenfield import probe_ws32_prefill_layer as probe
    from scripts.greenfield import prefill_router_protocol as router
    from scripts.greenfield import prefill_completed_window_admission as admission

    tree = ast.parse(Path(probe.__file__).read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    first = next(
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "prefix_mlp" for t in n.targets)
    )
    last = next(
        i
        for i, n in enumerate(main.body[first:], first)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "_atomic_json"
    )
    namespace = dict(
        vars(probe),
        tag=f"greenfield_fp8_{protocol.KERNEL}_l6_fixture",
        layer=6,
        args=SimpleNamespace(expected_code_hash="a" * 40, process_id=0),
        output=tmp_path / "runner.json",
        router_protocol=router,
    )
    inspected = []
    monkeypatch.setattr(
        admission, "registered_programs", lambda: inspected.append(True)
    )
    exec(
        compile(
            ast.Module(body=main.body[first : last + 1], type_ignores=[]),
            "<actual-probe-early-scope>",
            "exec",
        ),
        namespace,
    )
    assert inspected == [True]
    r = namespace["record"]
    assert (
        r["protocol"] == protocol.PROTOCOL
        and r["reference_scope"] == protocol.REFERENCE_SCOPE
    )
    assert r["independent_full_layer_admission"] is False and r["compile_only"] is False
    call = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "execute_acquisition"
    )
    observed = {}
    monkeypatch.setattr(
        acquisition, "execute_acquisition", lambda **kw: observed.update(kw)
    )
    slots = {9: 0, 13: 1, 25: 2, 29: 3}
    namespace.update(
        mesh=None, config=None, weights=None, consensus=None, local_slots=slots
    )
    eval(
        compile(ast.Expression(call), "<actual-probe-continuation>", "eval"), namespace
    )
    assert (
        observed["completed_window"] is True and observed["completed_numerical"] is True
    )
    assert observed["local_slots"] is slots
    assert (
        observed["boundary_diagnostic"] is False
        and observed["capture_boundaries"] is False
    )


def test_completed_fleet_original_publication_collector_and_database(
    tmp_path, monkeypatch
):
    pin, pins = "a" * 40, {"synthetic_metadata_fixture_only": True}
    ledger = {
        i: dict(selected={"tensor": str(i)}, full_sha256="d" * 64) for i in range(32)
    }
    order = [9, 13, 25, 29] + [i for i in range(32) if i not in (9, 13, 25, 29)]
    processes = [3, 5, 1, 2, 0, 6, 7, 4]
    monkeypatch.setattr(campaign, "checkpoint_ledger", lambda layer: (pins, ledger))
    source = tmp_path / "rank0"
    source.mkdir()
    with run_cases(source, create_journal=False) as (calls, sequence):
        continue_from_acquisition(source, calls, sequence)
        template = deepcopy(calls.record)
    records = []
    for rank in range(8):
        root = tmp_path / f"rank{rank}"
        slots = {order[s]: s for s in range(4 * rank, 4 * rank + 4)}
        r = (
            deepcopy(template)
            if rank == 0
            else clone_fixture(
                source, root, template, rank=rank, slots=slots, process=processes[rank]
            )
        )
        r.update(
            status="SUCCESS",
            hostname=f"fixture-host{rank}",
            pid=100 + rank,
            start_ticks=1234,
            boot_id=f"fixture-boot{rank}",
            layer=6,
            selected_layer_ids=[6],
            rows=128,
            control_rows=32,
            context_capacity=4096,
            key_tile=512,
            iterations=0,
            latency=None,
            admission_only=True,
            diagnostic_only=False,
            numerical_execution_authorized=True,
            state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
            integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
            checkpoint_pins=pins,
            payload_bytes_per_chip=window.PAYLOAD_BYTES_PER_CHIP,
            mesh_sha256=MESH_SHA,
            topology_sha256=TOPOLOGY_SHA,
            topology_fleet_sha256=FLEET_SHA,
            versions={"jax": "0.10.1", "libtpu": "0.0.41"},
            physical_device_ids=np.asarray(order).reshape(8, 4).tolist(),
        )
        for s in r["local_device_slots"]:
            slot = s["device_slot"]
            s.update(
                observed_selected_tensor_sha256=ledger[slot]["selected"],
                expected_full_file_sha256_not_verified=ledger[slot]["full_sha256"],
                selected_payload_bytes=window.PAYLOAD_BYTES_PER_CHIP,
            )
        preflight = dict(
            code_hash=pin,
            layer=6,
            launch_rank=rank,
            hostname=r["hostname"],
            headers=[dict(device_slot=s) for s in slots.values()],
        )
        raw = json.dumps(preflight).encode()
        (root / "retained_preflight.json").write_bytes(raw)
        r["retained_preflight_sha256"] = sha256(raw).hexdigest()
        (root / "runner.json").write_text(json.dumps(r))
        (root / "worker.log").write_text(
            "Synthetic CPU fleet capsule; NOT TPU evidence\n"
        )
        records.append(r)
    campaign.validate_workers(
        records, pin, layer=6, pins=pins, ledger=ledger, completed_numerical=True
    )
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication

    blobs = {}

    class Blob:
        generation, crc32c = 23, "fixture-crc"

        def __init__(self, name, data):
            self.name, self.data, self.size = name, data, len(data)

        def reload(self, *, if_generation_match):
            assert if_generation_match == self.generation

        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return self.data

    bucket = SimpleNamespace(
        get_blob=lambda name: blobs[name], blob=lambda name, generation: blobs[name]
    )

    def named_bucket(name):
        assert name == "driftbench-dsv4-uc"
        return bucket

    def publish(bucket, name, path, digest, *, compressed):
        assert not compressed
        data = path.read_bytes()
        b = Blob(name, data)
        blobs[name] = b
        return dict(
            name=name,
            generation=str(b.generation),
            size=b.size,
            crc32c=b.crc32c,
            original_sha256=sha256(data).hexdigest(),
        )

    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=named_bucket))
    monkeypatch.setattr(publication, "publish_exact", publish)
    tag = f"greenfield_fp8_{protocol.KERNEL}_l6_fixture"
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    for rank in range(8):
        campaign.publish_rank(tag, rank)
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path / "collected")
    result = campaign.collect(tag, pin)
    campaign.validate_record(result, pin)
    for mutate in (
        lambda r: r.update(independent_full_layer_admission=True),
        lambda r: r.update(performance_claim=True),
        lambda r: r["workers"][0].update(assembly_executable_calls=29),
        lambda r: r["workers"][0]["local_device_slots"][0].update(device_slot=4),
        lambda r: r["workers"][0]["call_evidence"][-1]["post_memory"][0].update(
            process_index=4
        ),
        lambda r: r["workers"][0]["local_device_slots"][0].update(
            observed_selected_tensor_sha256={}
        ),
        lambda r: r["workers"][0]["programs"]["assemble"].update(
            stablehlo_sha256="b" * 64
        ),
    ):
        bad = deepcopy(result)
        mutate(bad)
        with pytest.raises(ValueError):
            campaign.validate_record(bad, pin)
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    accounting = next(
        b.split("\nPY\n", 1)[0]
        for b in wrapper.split("<<'PY'\n")[1:]
        if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
    )
    (tmp_path / "runner.json").write_text(json.dumps(result))
    db = tmp_path / "test.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accounting",
            str(tmp_path),
            pin,
            str(db),
            str(Path.cwd()),
            "1",
            protocol.KERNEL,
        ],
    )
    exec(
        compile(accounting, "<actual-wrapper-accounting>", "exec"),
        {"__name__": "__main__"},
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "NOT independent full-layer DSA" in summary["claim_scope"]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "select item_id, correct, score, latency_ms from items"
        ).fetchall() == [
            (
                "layer6_completed_prefix_b128_four_b32_suffix_numerical_59calls_v1",
                1,
                1.0,
                None,
            )
        ]
