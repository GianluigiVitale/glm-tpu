from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

from safetensors import safe_open

from scripts.greenfield.pack_checkpoint_probe import pack_probe
from scripts.greenfield.load_checkpoint_probe import verify_probe_artifact
from tests.greenfield.checkpoint.test_stream_pack import fixture


REPO = Path(__file__).resolve().parents[3]


def test_checkpoint_probe_derives_small_complete_owner_files(
    tmp_path: Path,
) -> None:
    source_root, layout = fixture(tmp_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")
    selection = {
        "artifact_kind": "greenfield_checkpoint_probe_selection",
        "layout_manifest_sha256": layout["manifest_sha256"],
        "plan_group_hash": layout["plan_group_hash"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_names": ["model.layers.0.synthetic.weight"],
        "source_revision": layout["source"]["revision"],
        "topology_hash": layout["topology_hash"],
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "output"
    manifest = pack_probe(
        layout_path=layout_path,
        selection_path=selection_path,
        source_root=source_root,
        output_dir=output,
        code_hash="c" * 40,
    )

    assert manifest["file_count"] == 4
    assert manifest["selected_source_bytes"] == 128
    assert manifest["packed_payload_bytes"] == 128
    unhashed = dict(manifest); unhashed.pop("manifest_sha256")
    canonical = json.dumps(
        unhashed, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode()
    assert manifest["manifest_sha256"] == sha256(canonical).hexdigest()
    for record in manifest["files"]:
        path = output / record["filename"]
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert handle.keys() == ["model.layers.0.synthetic.weight"]
    evidence = (output / "evidence.sha256").read_text().splitlines()
    assert len(evidence) == 6
    verified_layout, verified_manifest, plans, by_filename = verify_probe_artifact(
        checkpoint_root=output,
        layout_path=layout_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    assert verified_layout["manifest_sha256"] == layout["manifest_sha256"]
    assert verified_manifest == manifest
    assert len(plans) == 4
    assert set(by_filename) == {plan.filename for plan in plans}


def test_pp16_probe_wrapper_is_bounded_same_region_and_terminal_last() -> None:
    wrapper = REPO / "scripts/greenfield/run_checkpoint_probe_pp16.sh"
    source = wrapper.read_text()
    for required in (
        "GLM_GREENFIELD_PP16_CHECKPOINT_PROBE",
        "APPROVED_BUCKET=gs://driftbench-dsv4-uc",
        "APPROVED_LOCATION=US-CENTRAL2",
        ".glm-tpu-rsync.lock",
        "JAX_PLATFORMS=cpu",
        "50345920",
        "50347904",
        "remote_objects.json",
        "generation",
        "crc32c",
        '"$REMOTE_PREFIX/SUCCESS"',
        "performance_claim':False",
    ):
        assert required in source
    for forbidden in (
        "driftbench-storage",
        "EUROPE-WEST4",
        "jax.distributed.initialize",
        "TPU_VISIBLE_DEVICES",
        "gcloud compute tpus",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(wrapper)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_pp16_probe_load_wrapper_has_census_db_hbm_and_terminal_contract() -> None:
    wrapper = REPO / "scripts/greenfield/run_checkpoint_probe_load_pp16.sh"
    source = wrapper.read_text()
    for required in (
        "GLM_GREENFIELD_PP16_CHECKPOINT_PROBE_LOAD",
        ".glm_pod_workload.lock",
        ".glm-tpu-rsync.lock",
        "strict_census pre",
        "strict_census post",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "load_checkpoint_probe.py",
        "device_roundtrip_bytes",
        "maximum_peak_hbm_bytes",
        "greenfield_checkpoint_probe_load_pp16",
        "results_ckpt.db",
        "remote_objects.json",
        '"$REMOTE_PREFIX/SUCCESS"',
        "performance_claim':False",
    ):
        assert required in source
    completed = subprocess.run(
        ["bash", "-n", str(wrapper)], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
