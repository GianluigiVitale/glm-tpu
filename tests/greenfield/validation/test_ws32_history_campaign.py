"""Existing probe/campaign plus real nested transport, with explicit CPU seams.

This tests orchestration and original-file plumbing, not repeated model math:
the independent original readers and selected-source provider have their own
byte-level suites. No actual SSH, cloud, checkpoint payload or TPU call.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

from google.cloud import storage
import numpy as np
import pytest

from scripts.greenfield import probe_ws32_prefill_layer as probe
from scripts.greenfield import ws32_history_campaign as glue
from scripts.greenfield import ws32_history_entry as entry
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from tests.greenfield.validation.test_ws32_history_transport import (
    Bucket, PIN, TAG, payloads, published, write,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def fixture(tmp_path, monkeypatch, published):
    root = tmp_path / TAG
    root.mkdir()
    monkeypatch.setattr(campaign, "run_root", lambda tag: root)
    monkeypatch.setattr(glue.preflight, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(glue.transport, "RUN_ROOT", tmp_path)
    bucket = Bucket()
    bucket.objects = deepcopy(published)
    monkeypatch.setattr(storage, "Client", bucket.client)
    write(root / "rank0", payloads(0))
    events = []
    original_root = root / "references"

    def originals(**kwargs):
        assert kwargs["root"] == root and kwargs["repo"] == REPO
        events.append("references")
        original_root.mkdir(exist_ok=True)
        return original_root
    monkeypatch.setattr(glue, "_references", originals)

    class Reader:
        receipt = dict(schema="fixture-source", sha256="d" * 64)

        def __call__(self, slot, name):
            assert (slot, name) == (0, "fixture.selected")
            events.append("selected_read")
            return np.array([3], np.uint8)

    def factory(mode, **kwargs):
        assert kwargs["root"] == root / "source_tiles" and kwargs["repo"] == REPO
        assert len(kwargs["records"]) == 8
        assert ("fetch" in kwargs) == (mode == "prepare")
        if mode == "prepare":
            assert kwargs["fetch"] is campaign.ssh
        events.append(mode)
        return Reader()

    # Fixture source transport, not synthetic claims presented as real payload.
    monkeypatch.setitem(sys.modules, "scripts.greenfield.ws32_history_source_reader",
        NS(prepare_reader=lambda **kw: factory("prepare", **kw),
           load_reader=lambda **kw: factory("load", **kw)))

    def replay(**kwargs):
        assert kwargs["root"] == root / "fleet" and kwargs["original_root"] == original_root
        assert kwargs["repo"] == REPO and len(kwargs["records"]) == 8
        assert (root / "fleet/rank7/call_records/call330.json").is_file()
        assert (root / "fleet/rank7/materializers/exact_promote.json").is_file()
        events.append("original_replay")
        assert kwargs["read_source"](0, "fixture.selected").tolist() == [3]
        return dict(reproduced=True, numerical_promotion=False, performance_claim=False)
    monkeypatch.setattr(glue.evidence, "validate_fleet", replay)
    return NS(root=root, bucket=bucket, events=events, reader=Reader)


def test_existing_collect_db_and_recovery_reuse_source_capsule(fixture):
    result = campaign.collect(TAG, PIN)
    assert fixture.events == ["references", "original_replay", "prepare", "selected_read"]
    assert result["diagnostic"]["selected_source_receipt"] == fixture.reader.receipt
    assert result["comparison"]["passed"] is None and result["latency"] is None
    campaign.validate_record(result, PIN)
    assert fixture.events[-3:] == ["original_replay", "load", "selected_read"]
    recovered = glue.replay_collected(tag=TAG, pin=PIN, root=fixture.root,
                                      repo=REPO, client=fixture.bucket.client())
    assert recovered == result
    assert fixture.events.count("prepare") == 1 and fixture.events.count("load") == 2
    with pytest.raises(FileExistsError):
        campaign.collect(TAG, PIN)


@pytest.mark.parametrize("mode", ["diagnostic", "materialized", "prefix_mlp", "observed"])
def test_history_cannot_be_relabelled_as_other_layer_mode(fixture, mode):
    with pytest.raises(ValueError, match="another layer mode"):
        campaign.validate_record(dict(kernel=glue.protocol.KERNEL), PIN, **{mode: True})
    assert fixture.events == []


def test_refused_original_identity_does_not_fetch_selected_payload(fixture, monkeypatch):
    def refuse(**kwargs):
        raise ValueError("original identity refused")
    monkeypatch.setattr(glue.evidence, "validate_fleet", refuse)
    with pytest.raises(ValueError, match="original identity"):
        campaign.collect(TAG, PIN)
    assert fixture.events == ["references"]
    assert (fixture.root / "fleet/rank7/call_records/call330.json").exists()


def test_unchecked_source_cannot_be_accepted(fixture, monkeypatch):
    monkeypatch.setattr(glue.evidence, "validate_fleet", lambda **kw:
                        dict(reproduced=True, numerical_promotion=False, performance_claim=False))
    with pytest.raises(ValueError, match="did not reconstruct"):
        campaign.collect(TAG, PIN)
    assert fixture.events == ["references"]


def test_probe_route_preserves_existing_early_guards(tmp_path, monkeypatch):
    args = NS(expected_code_hash=PIN, coordinator_address="127.0.0.1:8476",
              process_id=0, output_dir=tmp_path / TAG / "rank0")
    monkeypatch.setenv("GLM_GREENFIELD_RUN_TAG", TAG)
    monkeypatch.setattr(probe.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(probe, "_git_head", lambda: PIN)
    monkeypatch.setattr(probe, "Path", lambda s: tmp_path if s == "/home/gianl/glm-run" else Path(s))
    def forbidden(*a, **kw):
        raise AssertionError("history entered historical layer selection")
    monkeypatch.setattr(probe, "layer_from_tag", forbidden)
    called = []
    def execute(observed, *, tag, repo):
        assert observed is args and tag == TAG and repo == REPO
        assert args.num_processes == 8 and args.mesh_sha256 == probe.MESH_SHA
        called.append(True)
        (args.output_dir / "runner.json").write_text("fixture")
        return 0
    monkeypatch.setattr(entry, "execute", execute)
    assert probe.main() == 0 and called == [True]
    with pytest.raises(FileExistsError):
        probe.main()
    assert called == [True]


def test_original_reference_reuse_has_exact_rank0_hardlinks(tmp_path, monkeypatch):
    root = tmp_path / TAG
    source = root / "rank0/retained_reference"
    source.mkdir(parents=True)
    for branch in glue.protocol.BRANCHES:
        for form in glue.protocol.ORIGINAL_FORMS:
            (source / f"{branch}.{form}").write_bytes(f"{branch}.{form}".encode())
    events = []
    def load(path, **kwargs):
        assert path == source and kwargs == dict(repo=REPO, rank=0)
        events.append("authenticate")
    def materialize(path, **kwargs):
        rank = kwargs["rank"]
        assert path == root / f"references/rank{rank}/retained_reference"
        if rank == 0:
            assert events == ["authenticate"]
            for left in source.iterdir():
                right = path / left.name
                assert right.samefile(left) and right.read_bytes() == left.read_bytes()
        events.append(rank)
    monkeypatch.setattr(glue.preflight, "load_originals", load)
    monkeypatch.setattr(glue.preflight, "materialize_originals", materialize)
    assert glue._references(root=root, repo=REPO, client=object()) == root / "references"
    assert events == ["authenticate", *range(8)]


def test_existing_campaign_runs_all_preflights_before_worker_and_bounds_writes(fixture, monkeypatch):
    from scripts.greenfield import ws32_history_storage as bounded

    commands = []
    def ssh(command, *, output, **kwargs):
        commands.append((output.name, command, kwargs))
        if output.name == "fleet_sync.log":
            text = "".join(f"PREFILL_SYNC_OK host{i}\n" for i in range(8))
        elif output.name == "retained_preflight.log":
            text = "".join(f"PREFILL_RETAINED_OK host{i}\n" for i in range(8))
        elif output.name == "coordinator.log":
            text = "127.0.0.1\n"
        else:
            assert output.name == "fleet_launch.log" and kwargs["timeout"] == 1140
            assert "900s /home/gianl/vllm-env/bin/python" in command
            assert "180s /home/gianl/vllm-env/bin/python" in command
            assert "trap upload EXIT" in command
            assert "probe_ws32_prefill_layer.py" in command
            text = "fixture worker; bounded originals already published\n"
        bounded.write_bytes(output, text.encode())
    monkeypatch.setattr(campaign, "ssh", ssh)
    # run_logged has its own real child/limit tests; keep events in this process.
    sinks = []
    def logged(action, output):
        sinks.append(output.name)
        action()
    monkeypatch.setattr(bounded, "run_logged", logged)
    campaign.campaign(TAG, PIN)
    assert [name for name, _, _ in commands] == sinks == [
        "fleet_sync.log", "retained_preflight.log", "coordinator.log", "fleet_launch.log"]
    assert "JAX_PLATFORMS=cpu" in commands[1][1]
    assert "JAX_PLATFORMS=tpu" in commands[-1][1]
    assert (fixture.root / "hlo/candidate.optimized_hlo.txt").read_bytes() == (
        fixture.root / "fleet/rank0/candidate_b128.optimized_hlo.txt").read_bytes()
    assert json.loads((fixture.root / "runner.json").read_bytes())["kernel"] == glue.protocol.KERNEL


def test_history_compile_original_writer_refuses_before_disk_write(tmp_path, monkeypatch):
    from scripts.greenfield import ws32_history_worker_storage as bounded

    root = tmp_path / TAG / "rank0"
    root.mkdir(parents=True)
    monkeypatch.setattr(bounded, "MAX_HLO_BYTES", 10)
    record = dict(protocol=glue.protocol.PROTOCOL)
    probe._write_compiler_original(root, record, "candidate_b128", "stablehlo.mlir", "123456")
    first = root / "candidate_b128.stablehlo.mlir"
    with pytest.raises(ValueError, match="before write"):
        probe._write_compiler_original(root, record, "candidate_b128", "optimized_hlo.txt", "12345")
    assert first.read_text() == "123456"
    assert not (root / "candidate_b128.optimized_hlo.txt").exists()
    with pytest.raises(FileExistsError):
        probe._write_compiler_original(root, record, "candidate_b128", "stablehlo.mlir", "1")
    with pytest.raises(ValueError, match="identity"):
        probe._write_compiler_original(root, record, "unregistered", "stablehlo.mlir", "1")


def test_legacy_compile_writer_remains_unmodified(tmp_path):
    path = tmp_path / "historical.optimized_hlo.txt"
    probe._write_compiler_original(tmp_path, {}, "historical", "optimized_hlo.txt", "one")
    probe._write_compiler_original(tmp_path, {}, "historical", "optimized_hlo.txt", "two")
    assert path.read_text() == "two"
