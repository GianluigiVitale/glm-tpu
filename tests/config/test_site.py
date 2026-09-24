"""The site file is fail-closed: exact keys, owner-only regular file, validated values."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tomllib

import pytest

from glm_tpu.config import site as site_module
from glm_tpu.config.site import (
    SiteConfig,
    SiteConfigError,
    approved_source_uri,
    get_current_site,
    host_rank,
    rank_matches,
    set_current_site,
    to_toml,
)
from tests.fixtures.site import EXAMPLE_BUCKET, example_mapping, installed_site, write_example_site

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "site.example.toml"


@pytest.fixture(autouse=True)
def _no_site_environment(monkeypatch):
    for name in [k for k in os.environ if k.startswith("GLM_TPU_")]:
        monkeypatch.delenv(name)


def _write(tmp_path: Path, mapping: dict, name: str = "site.toml") -> Path:
    return write_example_site(tmp_path / name, mapping)


# ----------------------------------------------------------------------------- the example file
def test_example_file_is_a_complete_valid_site_once_placeholders_are_filled(tmp_path):
    text = EXAMPLE.read_text()
    placeholders = set(re.findall(r"<([a-z0-9-]+)>", text))
    assert placeholders, "the example documents its placeholders"
    values = {
        "64-hex": "0" * 64,
        "host0-internal-address": "203.0.113.10",
        "bucket": "example-bucket",
        "prefix": "models",
    }
    filled = re.sub(r"<([a-z0-9-]+)>", lambda m: values.get(m.group(1), "example-" + m.group(1)), text)
    site = SiteConfig.from_mapping(tomllib.loads(filled))
    assert site.fleet.num_hosts == 8 and site.fleet.chips_per_host == 4
    assert site.launch.allowed_branches == ("main", "release/*") and site.launch.expected_origin is None
    documented = tomllib.loads(filled)
    assert {k: set(v) for k, v in documented.items() if isinstance(v, dict)} == {
        name: set(spec) for name, spec in site_module._TABLES.items()
    }, "the example lists every key"


def test_example_file_has_only_neutral_values():
    text = EXAMPLE.read_text()
    assert not re.search(r"/home/|\b192\.168\.|\b10\.\d+\.\d+\.\d+|gs://(?!<bucket>/)", text)


# ----------------------------------------------------------------------------- loading
def test_load_validates_and_applies_defaults(tmp_path):
    mapping = example_mapping(tmp_path)
    for table, key in (
        ("fleet", "project"),
        ("fleet", "helper_python"),
        ("fleet", "known_hosts"),
        ("paths", "repo"),
        ("paths", "hlo_dump_root"),
    ):
        del mapping[table][key]
    del mapping["launch"]
    site = SiteConfig.load(_write(tmp_path, mapping))
    assert site.fleet.project == "" and site.fleet.helper_python == "python3"
    assert site.fleet.known_hosts == Path(os.path.expanduser("~/.ssh/google_compute_known_hosts"))
    assert site.paths.repo is None and site.paths.hlo_dump_root == Path("/dev/shm/glm-tpu-hlo")
    assert site.launch == site_module.LaunchPolicy()
    assert site.locks.workload == (tmp_path / "locks" / "lock0", tmp_path / "locks" / "lock1")


def test_site_file_location_follows_the_environment(tmp_path, monkeypatch):
    path = _write(tmp_path, example_mapping(tmp_path))
    monkeypatch.setenv("GLM_TPU_CONFIG_ROOT", str(tmp_path))
    assert SiteConfig.load().paths.run_root == tmp_path / "runs"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("GLM_TPU_SITE_CONFIG", str(_write(other, example_mapping(other))))
    assert SiteConfig.load().paths.run_root == other / "runs"
    assert SiteConfig.load(path).paths.run_root == tmp_path / "runs"  # the flag wins


def test_missing_site_file_names_the_example(tmp_path):
    with pytest.raises(SiteConfigError, match=r"site.example.toml"):
        SiteConfig.load(tmp_path / "absent.toml")


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o700, 0o660])
def test_site_file_must_be_owner_only(tmp_path, mode):
    path = _write(tmp_path, example_mapping(tmp_path))
    path.chmod(mode)
    with pytest.raises(SiteConfigError, match="mode 0600 or 0400"):
        SiteConfig.load(path)


def test_read_only_site_file_is_accepted(tmp_path):
    path = _write(tmp_path, example_mapping(tmp_path))
    path.chmod(0o400)
    assert SiteConfig.load(path).fleet.tpu_name == "example-vm"


def test_site_file_must_not_be_a_symlink(tmp_path):
    path = _write(tmp_path, example_mapping(tmp_path))
    link = tmp_path / "link.toml"
    link.symlink_to(path)
    with pytest.raises(SiteConfigError, match="not a symlink"):
        SiteConfig.load(link)


def test_invalid_toml_is_refused(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text("schema = [\n")
    path.chmod(0o600)
    with pytest.raises(SiteConfigError, match="not valid TOML"):
        SiteConfig.load(path)


def _refused(tmp_path, match: str, **changes):
    mapping = example_mapping(tmp_path)
    for dotted, value in changes.items():
        table, _, key = dotted.partition("__")
        if not key:
            mapping[table] = value
        elif value is KeyError:
            del mapping[table][key]
        else:
            mapping[table][key] = value
    with pytest.raises(SiteConfigError, match=match):
        SiteConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    "changes,match",
    [
        (dict(schema="glm_tpu_site_v0"), "schema"),
        (dict(fleet__tpu_name=KeyError), "missing site key fleet.tpu_name"),
        (dict(paths__run_root=KeyError), "missing site key paths.run_root"),
        (dict(checkpoint__manifest_sha256=KeyError), "missing site key checkpoint.manifest_sha256"),
        (dict(fleet__unknown="x"), "unknown site key fleet.unknown"),
        (dict(extra={}), "unknown site key 'extra'"),
        (dict(fleet__num_hosts=4), "fleet.num_hosts must be 8"),
        (dict(fleet__num_hosts=True), "fleet.num_hosts must be 8"),
        (dict(fleet__chips_per_host=8), "fleet.chips_per_host must be 4"),
        (dict(fleet__coordinator_address="203.0.113.10:8477"), "port must be 8476"),
        (dict(fleet__coordinator_address="host0.example.invalid:8476"), "IPv4 address"),
        (dict(fleet__host_rank_regex="-w-\\d+$"), "exactly one group"),
        (dict(fleet__host_rank_regex="-w-(\\d+"), "not a regular expression"),
        (dict(fleet__helper_python="python3 -S"), "bare command"),
        (dict(fleet__helper_python="../python3"), "bare command"),
        (dict(fleet__worker_python="bin/python3.12"), "absolute path"),
        (dict(fleet__worker_pythonpath="/opt/example/site-packages"), "list of strings"),
        (dict(fleet__tpu_name="bad name"), "invalid form"),
        (dict(paths__run_root="runs"), "absolute path"),
        (dict(paths__run_root="/opt/example/../runs"), "absolute path without '..'"),
        (dict(checkpoint__root="/elsewhere/pack"), "checkpoint.root must lie strictly inside"),
        (dict(checkpoint__manifest_sha256="A" * 64), "64-hex"),
        (dict(checkpoint__success_sha256="0" * 63), "64-hex"),
        (dict(topology__mesh_sha256="x"), "64-hex"),
        (dict(storage__source_uri="gs://other-bucket/models/x"), "under storage.allowed_source_uri_prefixes"),
        (dict(storage__allowed_source_uri_prefixes=["gs://example-bucket"]), "gs://<bucket>/"),
        (dict(storage__allowed_source_uri_prefixes=[]), "non-empty list"),
        (dict(locks__workload=[]), "non-empty list"),
        (dict(launch__allowed_branches=[]), "non-empty list"),
        (dict(launch__require_clean="yes"), "true or false"),
        (dict(launch__expected_origin=7), "string"),
    ],
)
def test_invalid_values_are_refused_naming_the_key(tmp_path, changes, match):
    _refused(tmp_path, match, **changes)


def test_checkpoint_children_must_be_strictly_inside(tmp_path):
    mapping = example_mapping(tmp_path)
    mapping["checkpoint"]["root"] = mapping["checkpoint"]["namespace"]
    with pytest.raises(SiteConfigError, match=r"strictly inside checkpoint.namespace"):
        SiteConfig.from_mapping(mapping)
    mapping = example_mapping(tmp_path)
    mapping["checkpoint"]["source_inventory"] = str(tmp_path / "elsewhere" / "inventory.json")
    with pytest.raises(SiteConfigError, match=r"strictly inside checkpoint.inventory_namespace"):
        SiteConfig.from_mapping(mapping)


def test_a_lock_cannot_be_both_workload_and_sync(tmp_path):
    mapping = example_mapping(tmp_path)
    mapping["locks"]["sync"] = [mapping["locks"]["workload"][0]]
    with pytest.raises(SiteConfigError, match="both workload and sync"):
        SiteConfig.from_mapping(mapping)


# ----------------------------------------------------------------------------- precedence
def test_environment_overrides_controller_paths_only_when_asked(tmp_path, monkeypatch):
    path = _write(tmp_path, example_mapping(tmp_path))
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", str(tmp_path / "other-runs"))
    monkeypatch.setenv("GLM_TPU_MODEL_PATH", str(tmp_path / "other-model"))
    monkeypatch.setenv("GLM_TPU_HLO_DUMP_ROOT", "/dev/shm/other-hlo")
    site = SiteConfig.load(path)
    assert site.paths.run_root == tmp_path / "other-runs"
    assert site.paths.model_path == tmp_path / "other-model"
    assert site.paths.hlo_dump_root == Path("/dev/shm/other-hlo")
    assert SiteConfig.load(path, environ=False).paths.run_root == tmp_path / "runs"
    monkeypatch.setenv("GLM_TPU_RUN_ROOT", "relative/runs")
    with pytest.raises(SiteConfigError, match=r"paths.run_root"):
        SiteConfig.load(path)


# ----------------------------------------------------------------------------- resolved form
def test_resolved_json_is_canonical_and_omits_controller_only_keys(tmp_path):
    site = SiteConfig.from_mapping(example_mapping(tmp_path))
    raw = site.resolved_json()
    value = json.loads(raw)
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["schema"] == "glm_tpu_site_resolved_v1" and "launch" not in value
    assert "known_hosts" not in value["fleet"] and "repo" not in value["paths"]
    worker = SiteConfig.from_resolved_json(raw)
    assert worker.launch is None and worker.paths.repo is None and worker.fleet.known_hosts is None
    assert worker.checkpoint == site.checkpoint and worker.locks == site.locks and worker.storage == site.storage
    assert worker.resolved_json() == raw and worker.resolved_sha256() == site.resolved_sha256()


def test_resolved_json_is_validated_like_the_file(tmp_path):
    site = SiteConfig.from_mapping(example_mapping(tmp_path))
    value = json.loads(site.resolved_json())
    with pytest.raises(SiteConfigError, match="canonical"):
        SiteConfig.from_resolved_json(json.dumps(value, indent=1).encode())
    for table, key, match in (
        ("fleet", "known_hosts", "unknown site key fleet.known_hosts"),
        ("paths", "repo", "unknown site key paths.repo"),
    ):
        changed = json.loads(site.resolved_json())
        changed[table][key] = "/x"
        with pytest.raises(SiteConfigError, match=match):
            SiteConfig.from_resolved_json(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode())
    changed = json.loads(site.resolved_json())
    del changed["fleet"]["helper_python"]  # a resolved site carries every key (no defaults)
    with pytest.raises(SiteConfigError, match=r"missing site key fleet.helper_python"):
        SiteConfig.from_resolved_json(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode())
    changed = json.loads(site.resolved_json())
    changed["launch"] = {}
    with pytest.raises(SiteConfigError, match="unknown site key 'launch'"):
        SiteConfig.from_resolved_json(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode())


def test_staged_site_is_bound_by_its_digest_and_owner_only(tmp_path):
    site = SiteConfig.from_mapping(example_mapping(tmp_path))
    path = tmp_path / "site.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(site.resolved_json())
    assert SiteConfig.from_staged(path, site.resolved_sha256()).storage == site.storage
    with pytest.raises(SiteConfigError, match="digest differs"):
        SiteConfig.from_staged(path, "0" * 64)
    path.chmod(0o644)
    with pytest.raises(SiteConfigError, match="mode 0600"):
        SiteConfig.from_staged(path, site.resolved_sha256())


def test_toml_writer_round_trips(tmp_path):
    mapping = example_mapping(tmp_path)
    assert tomllib.loads(to_toml(mapping)) == mapping


# ----------------------------------------------------------------------------- process site
def test_current_site_is_explicit_and_fail_closed(tmp_path):
    with installed_site(None):
        with pytest.raises(SiteConfigError, match="no site configuration is installed"):
            get_current_site()
        with pytest.raises(SiteConfigError):
            approved_source_uri(EXAMPLE_BUCKET + "models/x")
        site = SiteConfig.from_mapping(example_mapping(tmp_path))
        assert set_current_site(site) is None
        assert get_current_site() is site
        assert approved_source_uri(EXAMPLE_BUCKET + "models/x")
        assert not approved_source_uri("gs://other-bucket/models/x") and not approved_source_uri(None)
        assert set_current_site(None) is site
    with pytest.raises(TypeError):
        set_current_site(object())


def test_host_rank_follows_the_fleet_convention():
    assert host_rank("example-vm-w-3") == 3 and host_rank("example-vm") is None
    assert rank_matches("example-vm-w-3", 3) and not rank_matches("example-vm-w-13", 3)
    assert not rank_matches("example-vm-w-03", 3) and not rank_matches("example-w-3-x", 3)
    assert host_rank("node7", r"node(\d+)$") == 7


@pytest.mark.parametrize(
    "changes,match",
    [
        (dict(paths__run_root="/runs/with space"), "plain path"),
        (dict(paths__run_root="/runs/it's"), "plain path"),
        (dict(paths__run_root='/runs/"quoted"'), "plain path"),
        (dict(paths__model_path="/models/ünïcode"), "plain path"),
        (dict(paths__hlo_dump_root="/dev/shm/line\nbreak"), "plain path"),
        (dict(checkpoint__namespace="/dev/shm/tab\tname"), "plain path"),
        (dict(locks__workload=["/locks/a lock", "/locks/b"]), "plain path"),
        (dict(fleet__worker_pythonpath=["/opt/site packages"]), "plain path"),
        (dict(fleet__known_hosts="/opt/example/known hosts"), "plain path"),
        (dict(fleet__coordinator_address="::1:8476"), "IPv4"),
        (dict(fleet__coordinator_address="[::1]:8476"), "IPv4"),
        (dict(fleet__coordinator_address="203.0.113.010:8476"), "IPv4"),
        (dict(fleet__worker_pythonpath=[]), "non-empty list"),
        (dict(storage__allowed_source_uri_prefixes=["gs://example-bucket/../"]), "segments"),
        (dict(storage__allowed_source_uri_prefixes=["gs://example-bucket/./"]), "segments"),
        (dict(storage__source_uri="gs://example-bucket/models/../other"), "segments"),
        (dict(storage__source_uri="gs://example-bucket/models/with space"), "spaces"),
    ],
)
def test_unsafe_path_address_and_uri_values_are_refused(tmp_path, changes, match):
    _refused(tmp_path, match, **changes)


def test_rank_parsing_admits_plain_hostnames_only():
    assert host_rank("example-vm-w-3\n") is None and not rank_matches("example-vm-w-3\n", 3)
    assert host_rank("example vm-w-3") is None and host_rank("example-vm-w-٣") is None
    assert host_rank("example-vm-w-3.internal", r"-w-(\d+)\.internal$") == 3
