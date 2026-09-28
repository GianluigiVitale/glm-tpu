"""Tests of :mod:`glm_tpu.distributed.topology_capture`, the per-host capture process: the staged run directory is
authenticated before any device is opened, and the capture on a fake eight-host JAX runtime writes the record the
runtime authenticates. Never started as a child process: a process whose command line names a fleet module makes the
equivalence harness see a live TPU run."""

from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

from glm_tpu.config.site import SiteConfig, get_current_site, set_current_site
from glm_tpu.distributed import topology_capture as worker
from glm_tpu.distributed.topology import _TOPOLOGY_CAPTURE_KEYS, validate_topology_fleet
from glm_tpu.utils.json_utils import canonical
from tests.fixtures import topology as fleet
from tests.fixtures.site import example_mapping

PIN = "c" * 40
NAME = "topology_capture_20260101T000000000000Z"


@pytest.fixture(autouse=True)
def restore_site():
    try:
        previous = get_current_site()
    except ValueError:
        previous = None
    yield
    set_current_site(previous)


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A run directory as the job stages it on host 3: site.json and a source manifest of the real capture file."""
    runs = tmp_path / "runs"
    root = runs / NAME
    root.mkdir(parents=True, mode=0o700)
    runs.chmod(0o700)
    site = SiteConfig.from_mapping(example_mapping(tmp_path, paths=dict(run_root=str(runs))))
    (root / "site.json").write_bytes(site.resolved_json())
    manifest = canonical({worker.SELF: sha256((worker.REPO / worker.SELF).read_bytes()).hexdigest()}) + b"\n"
    (root / "source_manifest.json").write_bytes(manifest)
    for name in ("site.json", "source_manifest.json"):
        (root / name).chmod(0o600)
    monkeypatch.setenv(worker.ENV_FLAG, "1")
    monkeypatch.setattr(worker.socket, "gethostname", lambda: fleet.HOSTS[3])
    args = Namespace(
        output=root,
        code_hash=PIN,
        source_manifest_sha256=sha256(manifest).hexdigest(),
        site_sha256=site.resolved_sha256(),
        coordinator_address=site.fleet.coordinator_address,
    )
    return args, site


def test_preflight_names_the_rank_the_hostname_encodes(staged):
    args, site = staged
    staged_site, rank = worker.preflight(args)
    assert rank == 3 and staged_site.resolved_json() == site.resolved_json() and get_current_site() is staged_site


def _flag(args, monkeypatch):
    monkeypatch.delenv(worker.ENV_FLAG)


def _name(args, monkeypatch):
    other = args.output.with_name("topology_capture_1")
    args.output.rename(other)
    args.output = other


def _site_digest(args, monkeypatch):
    args.site_sha256 = "0" * 64


def _manifest_digest(args, monkeypatch):
    args.source_manifest_sha256 = "0" * 64


def _source_differs(args, monkeypatch):
    raw = canonical({worker.SELF: "0" * 64}) + b"\n"
    (args.output / "source_manifest.json").write_bytes(raw)
    args.source_manifest_sha256 = sha256(raw).hexdigest()


def _coordinator(args, monkeypatch):
    args.coordinator_address = "198.51.100.7:8476"


def _hostname(args, monkeypatch):
    monkeypatch.setattr(worker.socket, "gethostname", lambda: "example-host")


def _repeat(args, monkeypatch):
    (args.output / worker.capture_file(3)).write_text("{}")


def _pin(args, monkeypatch):
    args.code_hash = "main"


REFUSALS = {
    "flag": (_flag, "protected topology capture identity required"),
    "run name": (_name, "protected topology capture identity required"),
    "pin": (_pin, "protected topology capture identity required"),
    "site digest": (_site_digest, "staged site.json digest differs"),
    "manifest digest": (_manifest_digest, "source manifest digest differs"),
    "source": (_source_differs, "deployed capture source differs"),
    "coordinator": (_coordinator, "coordinator address differs"),
    "hostname": (_hostname, "capture rank differs"),
    "repeat": (_repeat, "never repeated"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_preflight_refuses(staged, monkeypatch: pytest.MonkeyPatch, case: str):
    args, _ = staged
    mutate, message = REFUSALS[case]
    mutate(args, monkeypatch)
    with pytest.raises(ValueError, match=message):
        worker.preflight(args)


def test_preflight_only_prints_this_hosts_facts_and_opens_no_device(staged, capsys: pytest.CaptureFixture[str]):
    args, _ = staged
    argv = ["--output", str(args.output), "--code-hash", PIN, "--source-manifest-sha256", args.source_manifest_sha256]
    argv += ["--site-sha256", args.site_sha256, "--coordinator-address", args.coordinator_address, "--preflight-only"]
    assert worker.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == dict(
        rank=3, hostname=fleet.HOSTS[3], code_hash=PIN, tpu_initialized=False
    )
    assert "jax" not in sys.modules or not isinstance(sys.modules["jax"], fleet.FakeJax)
    assert not (args.output / worker.capture_file(3)).exists()


def test_capture_writes_the_record_the_runtime_authenticates(staged, tmp_path: Path):
    _, site = staged
    (tmp_path / "all").mkdir()
    raws = fleet.capture_all(tmp_path / "all", site, PIN)
    records = [json.loads(raw) for raw in raws]
    assert all(set(record) == _TOPOLOGY_CAPTURE_KEYS for record in records)
    _, ordered, _ = validate_topology_fleet(
        tuple(records),
        expected_topology_sha256=records[0]["contract"]["topology_hash"],
        expected_fleet_sha256=_fleet_digest(records),
        slice_name=site.topology.slice_name,
    )
    assert [r["jax_process_index"] for r in ordered] == list(fleet.PERMUTATION)
    assert records[0]["contract"]["code_hash"] == PIN
    assert records[0]["contract"]["mesh_sha256"]
    assert oct((tmp_path / "all" / worker.capture_file(0)).stat().st_mode & 0o777) == "0o600"


def _fleet_digest(records):
    projection = dict(
        fleet_local_device_ids_in_runtime_order=records[0]["fleet_local_device_ids_in_runtime_order"],
        records=[
            {
                k: r[k]
                for k in ("contract_hash", "hostname", "jax_process_index", "launch_process_id", "local_device_ids")
            }
            for r in records
        ],
    )
    return sha256(canonical(projection)).hexdigest()


def test_capture_joins_the_fleet_as_its_rank_and_shuts_down(staged):
    args, site = staged
    with fleet.fake_jax(3) as jax:
        record = worker.capture(args, site, 3)
    assert jax.calls[0] == (
        "initialize",
        dict(coordinator_address=site.fleet.coordinator_address, num_processes=8, process_id=3),
    )
    assert jax.calls[-1] == ("shutdown",)
    assert [c[0] for c in jax.calls] == ["initialize", "sync", "shutdown"]
    assert record["launch_process_id"] == 3 and record["jax_process_index"] == fleet.PERMUTATION[3]
    assert json.loads((args.output / worker.capture_file(3)).read_text()) == record


def test_hosts_that_disagree_on_the_contract_write_nothing(staged):
    args, site = staged
    with fleet.fake_jax(3, digest_of={6: bytes(32)}) as jax, pytest.raises(RuntimeError, match="disagree"):
        worker.capture(args, site, 3)
    assert jax.calls[-1] == ("shutdown",)  # the runtime is left even on a refusal
    assert not (args.output / worker.capture_file(3)).exists()


def test_a_device_outside_the_gathered_ownership_refuses(staged):
    args, site = staged
    moved = [fleet.device(d, process_index=(d // 4 + 1) % 8 if d in (0, 1, 2, 3) else d // 4) for d in range(32)]
    with fleet.fake_jax(3, all_devices=moved), pytest.raises(Exception, match=r"process|chips per process"):
        worker.capture(args, site, 3)
    assert not (args.output / worker.capture_file(3)).exists()
