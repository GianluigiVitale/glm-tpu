"""Tests of :mod:`glm_tpu.executor.pack_job` (``glm-tpu checkpoint pack``) on the tiny checkpoint and a fake
eight-host fleet (``tests.fixtures.fleet``): every host packs, compares or installs with the pack worker's own
functions (``pack_runtime_slots``, ``pack_worker.compare_seal`` and ``pack_worker.install_seal``) in its own directory,
the controller assembles the manifest, seals it, stages the seal and has every host install and verify it; a kept
seal is installed only when every receipt equals it; the dry run compares tensors with a seal and writes no
checkpoint."""

from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import shlex

import pytest

from glm_tpu.config import model
from glm_tpu.distributed.topology import CAPTURE_NAMES, binding_bytes, derive_topology_binding
from glm_tpu.executor import pack_job
from glm_tpu.model_loader import pack_worker
from glm_tpu.model_loader.sharded_state.format import RuntimePackConfig, build_runtime_file_plans
from glm_tpu.model_loader.sharded_state.writer import pack_runtime_slots
from glm_tpu.model_loader.source_inventory import write_source_inventory
from tests.fixtures import fleet as fake
from tests.fixtures import tiny_checkpoint, topology
from tests.fixtures.site import example_mapping, installed_site

TAG = "greenfield_ws32_runtime_pack_" + "20000101T" + "0" * 15 + "Z"  # a synthetic pack run name


class Setup:
    """A site pinning the tiny source, its inventory and marker, a topology binding of the synthetic fleet, and a
    checkpoint root to pack; every host's checkpoint namespace is its own directory."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp = tmp_path
        runs, locks = tmp_path / "runs", tmp_path / "locks"
        runs.mkdir(mode=0o700)
        locks.mkdir()
        self.repo = tmp_path / "repo"
        self.pin = fake.synthetic_repo(self.repo)
        base = example_mapping(tmp_path)
        with installed_site(replace_site(base)):
            _, self.inventory, config = tiny_checkpoint.fixture(tmp_path)
        self.source = config.source_root
        (self.source / "SOURCE_COMPLETE.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    shards=[
                        dict(
                            name="model.safetensors",
                            sha256=sha256((self.source / "model.safetensors").read_bytes()).hexdigest(),
                        )
                    ],
                )
            )
        )
        inventory_path = tmp_path / "inventories" / "tiny" / "source_inventory.json"
        write_source_inventory(self.inventory, inventory_path)
        (tmp_path / "captures").mkdir()
        raws = topology.capture_all(tmp_path / "captures", replace_site(base), self.pin)
        self.binding = derive_topology_binding(raws, all_hosts_idle_after=True, note="test")
        binding_dir = tmp_path / "binding"
        (binding_dir / "captures").mkdir(parents=True)
        (binding_dir / "topology_rebinding.json").write_bytes(binding_bytes(self.binding))
        for name, raw in zip(CAPTURE_NAMES, raws, strict=True):
            (binding_dir / "captures" / name).write_bytes(raw)
        self.mapping = example_mapping(
            tmp_path,
            paths=dict(run_root=str(runs), model_path=str(self.source)),
            checkpoint=dict(
                namespace=str(tmp_path / "checkpoints"),
                root=str(tmp_path / "checkpoints" / TAG),
                inventory_namespace=str(tmp_path / "inventories"),
                source_inventory=str(inventory_path),
                source_inventory_sha256=self.inventory.inventory_sha256,
                source_complete_sha256=sha256((self.source / "SOURCE_COMPLETE.json").read_bytes()).hexdigest(),
            ),
            topology=dict(
                binding_dir=str(binding_dir),
                binding_sha256=sha256(binding_bytes(self.binding)).hexdigest(),
                capture_root=str(binding_dir / "captures"),
                topology_sha256=self.binding["original_topology_sha256"],
                topology_fleet_sha256=self.binding["original_fleet_sha256"],
                mesh_sha256=self.binding["mesh_sha256"],
            ),
        )
        # The tiny checkpoint stands in for GLM-5.3: its geometry, and no pinned-identity check of its inventory.
        monkeypatch.setattr(model, "geometry", lambda repo=None: tiny_checkpoint.geometry())
        monkeypatch.setattr(model, "require_inventory", lambda inventory: None)
        self.monkeypatch = monkeypatch
        self.parser = pack_worker_parser()  # once: the fake hosts answer in parallel threads

    def site(self, **checkpoint):
        return replace_site(self.mapping, checkpoint=checkpoint)

    def host_root(self, rank: int, tag: str) -> Path:
        return self.tmp / "checkpoints" / TAG if rank == 0 and tag == TAG else self.tmp / "hosts" / f"ckpt{rank}" / tag

    def plans(self):
        _, plans = build_runtime_file_plans(
            self.inventory, tiny_checkpoint.geometry(), mesh_hash=self.binding["mesh_sha256"]
        )
        return plans

    def args(self, fleet, rank, words):
        """The pack worker's arguments of a command (its flags), with this host's run directory as --output."""
        values, _ = self.parser.parse_known_args(words[words.index("-m") + 2 :])
        values.output = fleet.run_dir(rank)
        return values

    def run(self, site, *, tamper=None, fail_install=False, **options):
        slots = self.binding["host_to_slots"]

        def cpu(fleet, rank, words):
            if "--preflight-only" in words:
                facts = dict(
                    rank=rank,
                    hostname=fake.HOSTS[rank],
                    code_hash=self.pin,
                    slots=slots[str(rank)],
                    tpu_initialized=False,
                )
                return 0, (json.dumps(facts) + "\n").encode()
            assert "--install-seal" in words
            if fail_install and rank == 4:
                return 0, b'{"rank": 4, "verified": false}\n'
            args = self.args(fleet, rank, words)
            with installed_site(site):
                report = pack_worker.install_seal(
                    args,
                    self.inventory,
                    tiny_checkpoint.geometry(),
                    self.binding,
                    rank,
                    slots[str(rank)],
                    self.host_root(rank, fleet.root.name),
                    site,
                )
            return 0, (json.dumps(report) + "\n").encode()

        def start(fleet, rank, args):
            assert args["module"] == pack_job.MODULE and args["env"] == pack_job.ENV and args["pin"] == self.pin
            words = ["-m", pack_job.MODULE, *args["argv"]]
            if "--compare-seal" in words:
                values = self.args(fleet, rank, words)
                with installed_site(site):
                    pack_worker.compare_seal(
                        values, self.inventory, tiny_checkpoint.geometry(), rank, slots[str(rank)], self.plans(), site
                    )
                return 0
            config = RuntimePackConfig(
                source_root=self.source,
                source_uri=site.storage.source_uri,
                output_dir=self.host_root(rank, fleet.root.name),
                code_hash=self.pin,
                mesh_hash=self.binding["mesh_sha256"],
            )
            with installed_site(site):
                record = pack_runtime_slots(
                    config, self.inventory, tiny_checkpoint.geometry(), device_slots=slots[str(rank)]
                )
            if tamper is not None and rank == 6:
                record = tamper(record)
            receipt = dict(rank=rank, complete=True, owner_record=record)
            fleet.put(rank, f"packing.rank{rank}.json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
            return 0

        self.fleet = fake.FakeJobFleet(site.paths.run_root, cpu=cpu, start=start, hosts=self.tmp / "hosts")
        fake.install(self.monkeypatch, self.fleet, self.repo, self.pin)
        try:
            return pack_job.pack_checkpoint(site, **options)
        except Exception as exc:  # the outcome is what the test asserts
            return exc


def replace_site(mapping, **tables):
    from glm_tpu.config.site import SiteConfig

    merged = {name: (dict(value, **tables[name]) if name in tables else value) for name, value in mapping.items()}
    return SiteConfig.from_mapping(merged)


def pack_worker_parser():
    """The pack worker's own parser (its main builds it; taken where it would parse)."""
    import argparse

    captured = {}

    class Built(Exception):
        pass

    def capture(self, args=None, namespace=None):
        captured["parser"] = self
        raise Built

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        pack_worker.main([])
    except Built:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return Setup(tmp_path, monkeypatch)


def test_a_pack_seals_installs_and_verifies_every_host(setup, capsys):
    site = setup.site()
    report = setup.run(site)
    assert isinstance(report, dict), repr(report)
    fleet = setup.fleet
    assert report["mode"] == "pack" and report["passed"] is True and report["code_hash"] == setup.pin
    assert capsys.readouterr().out.splitlines()[0] == "RUN " + str(fleet.root) and fleet.root.name == TAG
    assert [r["verified"] for r in report["installed"]] == [True] * 8
    manifest_raw = (setup.host_root(0, TAG) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    success = json.loads((setup.host_root(0, TAG) / "SUCCESS").read_bytes())
    assert (
        report["manifest_sha256"] == manifest["manifest_sha256"]
        and report["success_sha256"] == success["success_sha256"]
    )
    assert report["manifest_file_sha256"] == sha256(manifest_raw).hexdigest() == success["manifest_file_sha256"]
    assert manifest["code_hash"] == setup.pin and success["tag"] == TAG
    assert (
        manifest["source"]["files"][0]["sha256"]
        == sha256((setup.source / "model.safetensors").read_bytes()).hexdigest()
    )
    for rank in range(1, 8):  # the same seal on every host
        assert (setup.host_root(rank, TAG) / "manifest.json").read_bytes() == manifest_raw
    for name, key in (
        ("pack_preflight.json", "remote_preflight_sha256"),
        ("pack_terminal.json", "remote_terminal_sha256"),
        ("pack_post_census.json", "post_census_sha256"),
    ):
        assert success[key] == sha256((fleet.root / name).read_bytes()).hexdigest()
    # the pinned pieces a worker checks: the seal staged, the preflight before the start, installs after the stage
    order = [(helper, "--install-seal" in (args if helper == "cpu" else [])) for helper, _, args in fleet.calls]
    assert order.index(("start_worker", False)) < order.index(("cpu", True))
    staged = [args["root"] for _, args in fleet.helper_calls("stage_bundle")]
    assert staged == [str(fleet.root)] * 8 + [str(fleet.root / "seal")] * 8


def test_recovery_installs_the_kept_seal_byte_for_byte(setup, tmp_path):
    first = setup.run(setup.site())
    kept = tmp_path / "kept"
    kept.mkdir()
    for name in ("manifest.json", "SUCCESS"):
        (kept / name).write_bytes((setup.host_root(0, TAG) / name).read_bytes())
    # the checkpoint is lost (tmpfs): a new namespace and run root, the same tag and pins
    import shutil

    shutil.rmtree(tmp_path / "checkpoints")
    shutil.rmtree(tmp_path / "hosts")
    setup.mapping["paths"]["run_root"] = str(tmp_path / "runs2")
    (tmp_path / "runs2").mkdir(mode=0o700)
    site = setup.site(manifest_sha256=first["manifest_sha256"], success_sha256=first["success_sha256"])
    report = setup.run(site, seal=kept)
    assert isinstance(report, dict), repr(report)
    assert report["mode"] == "recover" and report["manifest_sha256"] == first["manifest_sha256"]
    for rank in range(8):
        for name in ("manifest.json", "SUCCESS"):
            assert (setup.host_root(rank, TAG) / name).read_bytes() == (kept / name).read_bytes()
    assert json.loads((setup.fleet.root / "seal_compare.json").read_text()) == dict(problems=[], passed=True)


def test_recovery_refuses_receipts_that_differ_and_installs_nothing(setup, tmp_path):
    first = setup.run(setup.site())
    kept = tmp_path / "kept"
    kept.mkdir()
    for name in ("manifest.json", "SUCCESS"):
        (kept / name).write_bytes((setup.host_root(0, TAG) / name).read_bytes())
    import shutil

    shutil.rmtree(tmp_path / "checkpoints")
    shutil.rmtree(tmp_path / "hosts")
    setup.mapping["paths"]["run_root"] = str(tmp_path / "runs2")
    (tmp_path / "runs2").mkdir(mode=0o700)
    site = setup.site(manifest_sha256=first["manifest_sha256"], success_sha256=first["success_sha256"])

    def tamper(record):
        return dict(record, files=[dict(record["files"][0], crc32c="AAAAAA=="), *record["files"][1:]])

    outcome = setup.run(site, seal=kept, tamper=tamper)
    assert isinstance(outcome, ValueError) and "differ from the seal; nothing installed" in str(outcome)
    assert not any((setup.host_root(rank, TAG) / "manifest.json").exists() for rank in range(8))
    assert json.loads((setup.fleet.root / "seal_compare.json").read_text())["passed"] is False


def test_recovery_refuses_a_site_that_pins_another_seal(setup, tmp_path):
    setup.run(setup.site())
    kept = tmp_path / "kept"
    kept.mkdir()
    for name in ("manifest.json", "SUCCESS"):
        (kept / name).write_bytes((setup.host_root(0, TAG) / name).read_bytes())
    with pytest.raises(ValueError, match="not this seal's"):
        pack_job.pack_checkpoint(setup.site(), seal=kept)


def test_a_host_that_does_not_verify_fails_the_pack(setup):
    outcome = setup.run(setup.site(), fail_install=True)
    assert isinstance(outcome, ValueError) and "did not verify" in str(outcome)


def test_the_dry_run_compares_with_a_seal_and_writes_no_checkpoint(setup, tmp_path):
    setup.run(setup.site())
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "manifest.json").write_bytes((setup.host_root(0, TAG) / "manifest.json").read_bytes())
    report = setup.run(setup.site(), compare=kept, tensors=0)
    assert isinstance(report, dict), repr(report)
    assert report["passed"] is True and report["problems"] == []
    assert report["slots_compared"] == list(range(32)) and report["plans_compared"] == [32] * 8
    assert report["tensors_compared"] == report["tensors_equal"] == 32  # one tensor per tiny slot
    run_name = Path(report["run"]).name
    assert run_name != TAG and pack_job.TAG.fullmatch(run_name)
    assert not any(setup.host_root(rank, run_name).exists() for rank in range(8))


def test_the_dry_run_reports_a_tensor_that_differs(setup, tmp_path):
    setup.run(setup.site())
    kept = tmp_path / "kept"
    kept.mkdir()
    manifest = json.loads((setup.host_root(0, TAG) / "manifest.json").read_text())
    manifest["files"][9]["tensor_sha256"][0] = "0" * 64
    from glm_tpu.model_loader.sharded_state.format import mapping_hash

    manifest["manifest_sha256"] = mapping_hash(manifest, field="manifest_sha256")
    (kept / "manifest.json").write_text(json.dumps(manifest))
    report = setup.run(setup.site(), compare=kept)
    assert report["passed"] is False and report["tensors_equal"] == report["tensors_compared"] - 1
    assert report["problems"] == ["slot 9 tensor 0 (model.embed_tokens.weight) differs from the seal"]


def test_the_preflight_dry_run_starts_nothing(setup):
    report = setup.run(setup.site(), preflight_only=True)
    assert report["passed"] is True and [f["slots"] for f in report["facts"]] == [
        setup.binding["host_to_slots"][str(r)] for r in range(8)
    ]
    assert setup.fleet.helper_calls("start_worker") == []


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(seal=Path("a"), compare=Path("b")), "choose one"),
        (dict(wall_seconds=10), "600..86400"),
        (dict(tensors=-1), "0 \\(all\\) or positive"),
    ],
)
def test_pack_options_are_checked_before_any_host(setup, options, message):
    with pytest.raises(ValueError, match=message):
        pack_job.pack_checkpoint(setup.site(), **options)


def test_an_existing_checkpoint_root_is_never_repacked(setup, tmp_path):
    (tmp_path / "checkpoints" / TAG).mkdir(parents=True)
    with pytest.raises(ValueError, match="never repack"):
        pack_job.pack_checkpoint(setup.site())


def test_the_new_tag_has_the_seal_format():
    assert pack_job.TAG.fullmatch(pack_job.new_tag())


def test_worker_argv_names_every_pin(setup):
    site = setup.site()
    job = Namespace(site=site, root=Path("/runs/x"), pin=setup.pin)
    argv = pack_job.worker_argv(job, "m" * 64, "i" * 64)
    assert shlex.join(argv).count("--") == 9 and argv[argv.index("--code-hash") + 1] == setup.pin
