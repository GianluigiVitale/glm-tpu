"""Tests of :mod:`glm_tpu.entrypoints.cli.checkpoint`: ``glm-tpu checkpoint inventory`` and ``checkpoint verify`` on the
G4 tiny checkpoint (the one-tensor source and one-layer geometry of the checkpoint unit tests, which the G4 gate packs
too), CPU only. Each command prints what the checkpoint library returns and refuses exactly as the library does: the
exception's type and message as JSON on standard error, nothing on standard output, exit 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from glm_tpu.config import model
from glm_tpu.config.site import SiteConfig, get_current_site
from glm_tpu.entrypoints.cli.checkpoint import verify_checkpoint
from glm_tpu.entrypoints.cli.main import main
from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
from glm_tpu.model_loader.sharded_state.writer import pack_runtime_checkpoint
from glm_tpu.model_loader.source_inventory import (
    authenticated_inventory,
    inspect_source_inventory,
    read_source_inventory,
    write_source_inventory,
)
from tests.fixtures import tiny_checkpoint
from tests.fixtures.site import example_mapping, example_site, installed_site, write_example_site

REPO = Path(__file__).resolve().parents[3]
MESH, TOPOLOGY = "b" * 64, "c" * 64  # the pins the fixture packs and seals with
MODEL_ID, REVISION = "zai-org/GLM-5.3", "unit-fixture"
UNLOADED = ("jax", "jaxlib", "libtpu", "torch", "transformers")  # the model libraries (numpy hashes the tensors)


def run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    """``glm-tpu ARGV``: exit code, standard output, standard error."""
    capsys.readouterr()
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def refused(capsys: pytest.CaptureFixture[str], argv: list[str], expected: BaseException) -> None:
    """The command refuses with ``expected``, the exception the library raised for the same input."""
    code, out, err = run(capsys, argv)
    assert (code, out) == (1, "")
    assert json.loads(err) == dict(
        error=type(expected).__name__, message=str(expected), status=f"checkpoint {argv[1]} refused"
    )


# ------------------------------------------------------------------------------------------------ checkpoint verify
@dataclass
class Packed:
    root: Path  # the sealed checkpoint, all 32 files
    mapping: dict[str, Any]  # the tables of a site file pinning it
    site_file: Path  # that site file (owner-only)


@pytest.fixture
def packed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Packed:
    """The G4 tiny checkpoint packed and sealed under ``tmp_path``, its source inventory written and a site file pinning
    both; the command's geometry is the tiny one (verify_checkpoint's keyword default: the pinned GLM-5.3 one)."""
    with installed_site(example_site(tmp_path / "pack")):  # the packer admits the fixture's example source URI
        _, inventory, config = tiny_checkpoint.fixture(tmp_path)
        manifest = pack_runtime_checkpoint(config, inventory, tiny_checkpoint.geometry(), chunk_bytes=16)
    success = tiny_checkpoint.seal(config.output_dir, manifest, topology_hash=TOPOLOGY)
    inventory_path = tmp_path / "inventories" / "tiny" / "source_inventory.json"
    write_source_inventory(inventory, inventory_path)
    mapping = example_mapping(
        tmp_path,
        checkpoint=dict(
            namespace=str(tmp_path),
            root=str(config.output_dir),
            inventory_namespace=str(tmp_path / "inventories"),
            source_inventory=str(inventory_path),
            source_inventory_sha256=inventory.inventory_sha256,
            manifest_sha256=manifest["manifest_sha256"],
            success_sha256=success["success_sha256"],
        ),
        topology=dict(mesh_sha256=MESH, topology_sha256=TOPOLOGY),
    )
    monkeypatch.setitem(verify_checkpoint.__kwdefaults__, "geometry", tiny_checkpoint.geometry())
    return Packed(config.output_dir, mapping, write_example_site(tmp_path / "site.toml", mapping))


def library(mapping: dict[str, Any], root: Path | None = None, geometry: Any = None, **options: Any) -> Any:
    """The library calls the command makes, spelled out: the site's source inventory against its pin, then the
    verification of the root with the site's pins and every file hashed unless ``options`` name the slots."""
    checkpoint, topology = mapping["checkpoint"], mapping["topology"]
    with installed_site(SiteConfig.from_mapping(mapping)):
        inventory = authenticated_inventory(Path(checkpoint["source_inventory"]), checkpoint["source_inventory_sha256"])
        return verify_runtime_checkpoint(
            Path(checkpoint["root"]) if root is None else root,
            expected_manifest_sha256=checkpoint["manifest_sha256"],
            expected_success_sha256=checkpoint["success_sha256"],
            expected_mesh_hash=topology["mesh_sha256"],
            expected_topology_hash=topology["topology_sha256"],
            inventory=inventory,
            geometry=tiny_checkpoint.geometry() if geometry is None else geometry,
            verify_file_hashes=True,
            **options,
        )


def local_root(packed: Packed, slots: tuple[int, ...]) -> Path:
    """A worker's local layout: the manifest, the seal and the files of ``slots`` only."""
    local = packed.root.parent / "local"
    local.mkdir()
    for name in ("manifest.json", "SUCCESS", *(f"device_slot_{slot:02d}.safetensors" for slot in slots)):
        (local / name).write_bytes((packed.root / name).read_bytes())
    return local


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def test_verify_prints_what_the_library_verified(packed: Packed, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    verified = library(packed.mapping)
    sentinel = example_site(tmp_path / "other")
    with installed_site(sentinel):
        code, out, err = run(capsys, ["checkpoint", "verify", "--site", str(packed.site_file)])
        assert get_current_site() is sentinel  # the site the command installed for the call is uninstalled again
    assert (code, err) == (0, "")
    report = json.loads(out)
    assert out == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert report == dict(
        schema="glm_tpu_checkpoint_verification_v1",
        root=str(packed.root),
        manifest_sha256=verified.manifest["manifest_sha256"],
        success_sha256=verified.success["success_sha256"],
        source_inventory_sha256=packed.mapping["checkpoint"]["source_inventory_sha256"],
        mesh_sha256=MESH,
        topology_sha256=TOPOLOGY,
        files=len(verified.plans),
        hashed_slots=list(range(32)),
        local_slot_layout=False,
        passed=True,
        scope=report["scope"],
    )
    assert report["files"] == 32
    assert report["scope"].endswith("no TPU, fleet, lock or remote host")


def test_verify_a_workers_local_layout(packed: Packed, capsys: pytest.CaptureFixture[str]):
    local = local_root(packed, (0, 1, 2, 3))
    library(packed.mapping, root=local, verify_file_hash_slots=(0, 1, 2, 3), local_slot_layout=True)
    argv = ["checkpoint", "verify", "--site", str(packed.site_file), "--root", str(local)]
    code, out, err = run(capsys, [*argv, "--slots", "3", "1", "0", "2", "--local-slot-layout"])
    assert (code, err) == (0, "")
    report = json.loads(out)
    assert (report["root"], report["hashed_slots"], report["local_slot_layout"]) == (str(local), [0, 1, 2, 3], True)


def test_verify_hashes_only_the_slots_given(packed: Packed, capsys: pytest.CaptureFixture[str]):
    """As the library: a slot that is not hashed is checked by size only (each host hashes its own slots)."""
    flip_last_byte(packed.root / "device_slot_31.safetensors")
    library(packed.mapping, verify_file_hash_slots=(0, 1, 2, 3))
    code, out, err = run(
        capsys, ["checkpoint", "verify", "--site", str(packed.site_file), "--slots", "0", "1", "2", "3"]
    )
    assert (code, err, json.loads(out)["hashed_slots"]) == (0, "", [0, 1, 2, 3])


def _flip_slot_31(root: Path) -> None:
    flip_last_byte(root / "device_slot_31.safetensors")


def _remove_slot_5(root: Path) -> None:
    (root / "device_slot_05.safetensors").unlink()


def _remove_success(root: Path) -> None:
    (root / "SUCCESS").unlink()


def _manifest_not_json(root: Path) -> None:
    (root / "manifest.json").write_text("not json")


# case: (site-file changes by table, command options, library options, tampering of the checkpoint root)
REFUSALS: dict[str, tuple[dict[str, dict[str, str]], list[str], dict[str, Any], Any]] = {
    "manifest pin": (dict(checkpoint=dict(manifest_sha256="0" * 64)), [], {}, None),
    "success pin": (dict(checkpoint=dict(success_sha256="0" * 64)), [], {}, None),
    "mesh pin": (dict(topology=dict(mesh_sha256="0" * 64)), [], {}, None),
    "topology pin": (dict(topology=dict(topology_sha256="0" * 64)), [], {}, None),
    "inventory pin": (dict(checkpoint=dict(source_inventory_sha256="0" * 64)), [], {}, None),
    "payload byte": ({}, [], {}, _flip_slot_31),
    "hashed slot payload byte": ({}, ["--slots", "31"], dict(verify_file_hash_slots=(31,)), _flip_slot_31),
    "missing file": ({}, [], {}, _remove_slot_5),
    "no SUCCESS": ({}, [], {}, _remove_success),
    "manifest not JSON": ({}, [], {}, _manifest_not_json),
    "local layout without slots": ({}, ["--local-slot-layout"], dict(local_slot_layout=True), None),
    "duplicate slot": ({}, ["--slots", "0", "0"], dict(verify_file_hash_slots=(0, 0)), None),
    "slot 32": ({}, ["--slots", "32"], dict(verify_file_hash_slots=(32,)), None),
    "foreign slots in a local layout": (
        {},
        ["--slots", "0", "1", "2", "3", "--local-slot-layout"],
        dict(verify_file_hash_slots=(0, 1, 2, 3), local_slot_layout=True),
        None,
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_verify_refuses_exactly_as_the_library(packed: Packed, capsys: pytest.CaptureFixture[str], case: str):
    changes, arguments, options, tamper = REFUSALS[case]
    mapping = {
        table: dict(values, **changes.get(table, {})) if isinstance(values, dict) else values
        for table, values in packed.mapping.items()
    }
    if tamper is not None:
        tamper(packed.root)
    with pytest.raises((ValueError, OSError)) as raised:
        library(mapping, **options)
    site_file = write_example_site(packed.root.parent / "refused-site.toml", mapping)
    refused(capsys, ["checkpoint", "verify", "--site", str(site_file), *arguments], raised.value)


def test_verify_uses_the_pinned_glm53_geometry_by_default(
    packed: Packed, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Without the tiny geometry the command re-derives the layout from GLM-5.3's, which this checkpoint is not."""
    monkeypatch.setitem(verify_checkpoint.__kwdefaults__, "geometry", None)
    with pytest.raises((ValueError, OSError)) as raised:
        library(packed.mapping, geometry=model.geometry())
    refused(capsys, ["checkpoint", "verify", "--site", str(packed.site_file)], raised.value)


@pytest.mark.parametrize("given", [True, False], ids=["--site", "default"])
def test_verify_refuses_a_missing_site_file_as_the_site_loader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], given: bool
):
    path = tmp_path / "absent.toml" if given else None  # the test session's default location does not exist
    with pytest.raises(ValueError) as raised:
        SiteConfig.load(path)
    refused(capsys, ["checkpoint", "verify", *(["--site", str(path)] if given else [])], raised.value)


def test_verify_in_a_fresh_process_loads_no_model_library(packed: Packed):
    geometry = tiny_checkpoint.geometry().to_dict()
    code = f"""
import json, sys
from glm_tpu.config.model import ModelGeometry
from glm_tpu.entrypoints.cli.checkpoint import verify_checkpoint
from glm_tpu.entrypoints.cli.main import main
value = {geometry!r}
value.update((k, tuple(value[k])) for k in ("fp8_block_shape", "mlp_layer_types", "indexer_types"))
verify_checkpoint.__kwdefaults__["geometry"] = ModelGeometry.from_dict(value)
assert main(["checkpoint", "verify", "--site", {str(packed.site_file)!r}]) == 0
print(json.dumps([name for name in {UNLOADED!r} if name in sys.modules]))
"""
    environment = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []


# --------------------------------------------------------------------------------------------- checkpoint inventory
@pytest.fixture
def source(tmp_path: Path) -> Path:
    """The G4 tiny checkpoint's source directory (index and one shard) with a config.json."""
    with installed_site(example_site(tmp_path / "pack")):
        _, _, config = tiny_checkpoint.fixture(tmp_path)
    (config.source_root / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    return config.source_root


def inventory_argv(source: Path, output: Path) -> list[str]:
    return [
        "checkpoint",
        "inventory",
        str(source),
        "--output",
        str(output),
        "--model-id",
        MODEL_ID,
        "--revision",
        REVISION,
    ]


def test_inventory_writes_what_the_library_reads(source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "inventories" / "source_inventory.json"
    code, out, err = run(capsys, inventory_argv(source, output))
    assert (code, err) == (0, "")
    expected = read_source_inventory(source, model_id=MODEL_ID, source_revision=REVISION)
    reference = tmp_path / "reference.json"
    write_source_inventory(expected, reference)
    assert output.read_bytes() == reference.read_bytes()
    assert inspect_source_inventory(output) == expected
    report = dict(
        schema="glm_tpu_source_inventory_report_v1",
        output=str(output),
        model_id=MODEL_ID,
        source_revision=REVISION,
        index_sha256=expected.index_sha256,
        config_sha256=expected.config_sha256,
        **expected.summary_dict(),
    )
    assert out == json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert not list(output.parent.glob(".*.tmp"))


def _existing_output(source: Path, output: Path) -> None:
    output.write_text("earlier inventory\n")


def _no_config(source: Path, output: Path) -> None:
    (source / "config.json").unlink()


def _no_index(source: Path, output: Path) -> None:
    (source / "model.safetensors.index.json").unlink()


def _index_names_a_missing_tensor(source: Path, output: Path) -> None:
    index = source / "model.safetensors.index.json"
    value = json.loads(index.read_text())
    value["weight_map"]["model.norm.weight"] = "model.safetensors"
    index.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "tamper", [_existing_output, _no_config, _no_index, _index_names_a_missing_tensor], ids=lambda f: f.__name__[1:]
)
def test_inventory_refuses_exactly_as_the_library(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], tamper: Any
):
    output = tmp_path / "source_inventory.json"
    tamper(source, output)
    before = output.read_bytes() if output.exists() else None
    with pytest.raises((ValueError, OSError)) as raised:
        write_source_inventory(read_source_inventory(source, model_id=MODEL_ID, source_revision=REVISION), output)
    refused(capsys, inventory_argv(source, output), raised.value)
    assert (output.read_bytes() if output.exists() else None) == before  # nothing written, nothing overwritten


def test_inventory_refuses_a_file_that_does_not_read_back_as_written(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    from glm_tpu.model_loader import source_inventory

    real, read = source_inventory.inspect_source_inventory, []

    def other(path: Path):  # the written file parses as another inventory
        read.append(path)
        return replace(real(path), source_revision="other")

    output = tmp_path / "source_inventory.json"
    monkeypatch.setattr(source_inventory, "inspect_source_inventory", other)
    code, out, err = run(capsys, inventory_argv(source, output))
    assert (code, out) == (1, "") and read == [output]
    assert json.loads(err) == dict(
        error="CheckpointValidationError",
        message=f"source inventory {output} does not read back as written",
        status="checkpoint inventory refused",
    )


def test_inventory_in_a_fresh_process_loads_no_model_library(source: Path, tmp_path: Path):
    # the inventory reads headers only: not even numpy (verify hashes tensors with it)
    output = tmp_path / "source_inventory.json"
    code = f"""
import json, sys
from glm_tpu.entrypoints.cli.main import main
assert main({inventory_argv(source, output)!r}) == 0
print(json.dumps([name for name in {(*UNLOADED, "numpy")!r} if name in sys.modules]))
"""
    environment = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []
    assert inspect_source_inventory(output) == read_source_inventory(
        source, model_id=MODEL_ID, source_revision=REVISION
    )
