"""Actual DB604/605 originals and selected metadata; fixture cloud, never TPU."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_dense_norm_originals as norm
from scripts.greenfield import ws32_dense_norm_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_protocol as prior_protocol
from scripts.greenfield import ws32_dense_frontier_preflight as preflight
from tests.greenfield.validation.test_ws32_dense_frontier_preflight import (
    fake_cloud as prior_cloud,
)

REPO = Path(__file__).resolve().parents[3]
ROOT = Path("/home/gianl/glm-run") / norm.TAG / "fleet"
PRIOR = (
    Path("/home/gianl/glm-run") / prior_protocol.ORIGINAL_TAG / "first_window_collected"
)
TAG = "greenfield_fp8_ws32_dense_norm_d01_20260909T170000000000000Z"


def fake_cloud(rank):
    client, bucket, blobs, events = prior_cloud(rank)
    root = ROOT / f"rank{rank}"
    for relative, pin in norm.reference_pins(root, repo=REPO, rank=rank).items():

        class Blob:
            def reload(self, *, if_generation_match):
                assert if_generation_match == int(self.generation)

            def download_to_filename(self, name, *, if_generation_match):
                assert if_generation_match == int(self.generation)
                events.append(self.name)
                Path(name).write_bytes(self.raw)

        value = Blob()
        vars(value).update(
            raw=(root / relative).read_bytes(),
            name=pin["name"],
            generation=pin["generation"],
            size=pin["size"],
            crc32c=pin["crc32c"],
        )
        blobs[pin["name"]] = value
    return client, bucket, blobs, events


def test_seven_exact_originals_and_reattach_no_overwrite(tmp_path):
    client, _, _, events = fake_cloud(0)
    root = tmp_path / "reference"
    runner, arrays, identity = norm.materialize(root, repo=REPO, rank=0, client=client)
    assert len(events) == 7 and len(arrays) == 5
    assert identity["raw_array_bytes"] == 81933312
    assert identity["bytes"] == sum(p.stat().st_size for p in root.iterdir())
    assert {v["device_slot"] for v in runner["local_device_slots"]} == {9, 13, 25, 29}
    norm.materialize(root, repo=REPO, rank=0, client=client)
    assert len(events) == 7
    original = root / "wide_final.npz"
    original.write_bytes(b"bad original")
    with pytest.raises(ValueError):
        norm.materialize(root, repo=REPO, rank=0, client=client)
    assert original.read_bytes() == b"bad original" and len(events) == 7


@pytest.mark.parametrize("kind", ["partial", "dangling", "target_exists"])
def test_partial_original_or_symlink_refuses_without_download(tmp_path, kind):
    client, _, _, events = fake_cloud(0)
    root = tmp_path / "reference"
    root.mkdir()
    partial = root / "worker_receipts.json.partial"
    target = tmp_path / "not_evidence"
    if kind == "partial":
        partial.write_bytes(b"partial original")
    else:
        partial.symlink_to(target)
        if kind == "target_exists":
            target.write_bytes(b"keep target")
    with pytest.raises(FileExistsError):
        norm.materialize(root, repo=REPO, rank=0, client=client)
    assert not events
    if kind == "partial":
        assert partial.read_bytes() == b"partial original"
    elif kind == "target_exists":
        assert target.read_bytes() == b"keep target"
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    "failure", ["region", "crc", "size", "bytes", "disk", "combined"]
)
def test_transport_refuses_before_unneeded_payloads(tmp_path, monkeypatch, failure):
    client, bucket, blobs, events = fake_cloud(0)
    pin = norm.receipt_worker(REPO, 0)["source_ledger"]
    first = blobs[pin["name"]]
    if failure == "region":
        bucket.location = "EU"
    elif failure == "crc":
        first.crc32c = "AAAAAA=="
    elif failure == "size":
        first.size += 1
    elif failure == "bytes":
        first.raw = b"X" + first.raw[1:]
    elif failure == "disk":
        monkeypatch.setattr(norm.shutil, "disk_usage", lambda path: NS(free=0))
    with pytest.raises((ValueError, SystemExit)):
        norm.materialize(
            tmp_path / "reference",
            repo=REPO,
            rank=0,
            client=client,
            existing_reference_bytes=(
                norm.MAX_REFERENCE_BYTES if failure == "combined" else 0
            ),
        )
    assert len(events) == (1 if failure in ("bytes", "combined") else 0)


def test_all_actual_rank_bindings_and_combined_storage_cap():
    amounts = []
    for rank in range(8):
        root = ROOT / f"rank{rank}"
        runner = json.loads((root / "runner.json").read_bytes())
        raw = (PRIOR / prior_protocol.original_names(rank)[0]).read_bytes()
        prior = json.loads(raw)
        norm.bind_prior(runner, prior, prior_sha256=sha256(raw).hexdigest())
        amount = sum(
            v["size"] for v in norm.reference_pins(root, repo=REPO, rank=rank).values()
        )
        amount += prior_protocol.LEDGER_PIN["size"] + sum(
            v["size"] for v in prior_protocol.reference_pins(PRIOR, rank).values()
        )
        assert amount < norm.MAX_REFERENCE_BYTES
        amounts.append(amount)
    print("ACTUAL_DB604_DB605_REFERENCE_BYTES_BY_RANK", amounts)


@pytest.mark.parametrize(
    "field",
    [
        "hostname",
        "jax_process_index",
        "checkpoint_manifest_sha256",
        "prompt_ids_sha256",
        "original_runner_sha256",
        "device_id",
        "file",
        "payload",
        "selected",
        "duplicate",
    ],
)
def test_wrong_original_context_or_selected_owners_refuse(field):
    runner = json.loads((ROOT / "rank0/runner.json").read_bytes())
    raw = (PRIOR / prior_protocol.original_names(0)[0]).read_bytes()
    if field == "device_id":
        runner["local_device_slots"][0]["device_id"] = 99
    elif field == "file":
        runner["local_device_slots"][0]["expected_full_file_sha256_not_verified"] = (
            "0" * 64
        )
    elif field == "payload":
        runner["local_device_slots"][0]["selected_payload_bytes"] += 1
    elif field == "selected":
        runner["local_device_slots"][0]["observed_selected_tensor_sha256"].popitem()
    elif field == "duplicate":
        runner["local_device_slots"][-1] = deepcopy(runner["local_device_slots"][0])
    else:
        runner[field] = "wrong"
    with pytest.raises(ValueError):
        norm.bind_prior(runner, json.loads(raw), prior_sha256=sha256(raw).hexdigest())


def test_actual_preflight_both_sources_and_headers_before_runtime(
    tmp_path, monkeypatch
):
    import jax

    def forbidden(*args, **kwargs):
        pytest.fail("preflight initialized devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "device_put", forbidden)
    monkeypatch.setattr(preflight, "RUN_ROOT", tmp_path)
    client, _, _, events = fake_cloud(0)
    root = tmp_path / TAG / "rank0"
    preflight.retained_preflight(
        tag=TAG, rank=0, pin="a" * 40, root=root, repo=REPO, client=client
    )
    report = json.loads((root / "retained_preflight.json").read_bytes())
    assert len(events) == 13 and report["protocol"] == protocol.PROTOCOL
    assert report["norm_originals"]["tag"] == norm.TAG
    assert report["combined_reference_bytes"] < norm.MAX_REFERENCE_BYTES
    assert report["norm_originals"]["raw_array_bytes"] == 81933312
    assert report["selected_leaf_count"] == 55 and report["include_embedding"]
    with pytest.raises(FileExistsError):
        preflight.retained_preflight(
            tag=TAG, rank=0, pin="a" * 40, root=root, repo=REPO, client=client
        )
