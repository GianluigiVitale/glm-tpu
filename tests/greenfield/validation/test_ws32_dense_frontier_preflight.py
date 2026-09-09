"""Actual retained DB604 originals and production metadata; no TPU/payload load."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_preflight as preflight
from scripts.greenfield.ws32_dense_frontier_witness import load_witness

REPO = Path(__file__).resolve().parents[3]
ORIGINAL = Path("/home/gianl/glm-run") / protocol.ORIGINAL_TAG / "first_window_collected"
TAG = "greenfield_fp8_ws32_dense_frontier_d01_20260909T140000000000000Z"


@pytest.fixture(scope="module")
def originals():
    # These are real saved TPU outputs, not manufactured comparison data.
    assert (ORIGINAL / "sources.json").exists(), "requires original DB604 evidence"
    return {rank: protocol.load_reference(ORIGINAL, rank=rank) for rank in range(8)}


def fake_cloud(rank):
    """Transport fixture serves the exact original SHA/CRC/generation payloads."""
    pins = {"sources.json": protocol.LEDGER_PIN, **protocol.reference_pins(ORIGINAL, rank)}
    events, blobs = [], {}
    for relative, pin in pins.items():
        raw = (ORIGINAL / relative).read_bytes()
        class Blob:
            def reload(self, *, if_generation_match):
                assert if_generation_match == int(self.generation)
            def download_to_filename(self, name, *, if_generation_match):
                assert if_generation_match == int(self.generation)
                events.append(self.name)
                Path(name).write_bytes(self.raw)
        blobs[pin["name"]] = Blob()
        vars(blobs[pin["name"]]).update(raw=raw, name=pin["name"], generation=pin["generation"],
                                       size=pin["size"], crc32c=pin["crc32c"])
    def blob(name, *, generation):
        value = blobs[name]
        assert generation == int(value.generation)
        return value
    bucket = NS(location="US-CENTRAL2", reload=lambda: None, blob=blob)
    def get_bucket(name):
        assert name == protocol.BUCKET
        return bucket
    return NS(bucket=get_bucket), bucket, blobs, events


def test_each_rank_reads_its_real_owners_not_rank_times_four(originals):
    whole = load_witness(ORIGINAL)
    union = {}
    for rank, (record, witness) in originals.items():
        assert len(witness) == 2 * 4 * 3
        assert record["launch_process_id"] == rank
        assert not union.keys() & witness.keys()
        union.update(witness)
        for key, value in witness.items():
            assert value["cache"] == whole[key]["cache"]
            assert value["rows"].tobytes() == whole[key]["rows"].tobytes()
    assert union.keys() == whole.keys()
    assert {k[1] for k in originals[0][1]} == {9, 13, 25, 29}


@pytest.mark.parametrize("rank", [True, -1, 8, "0"])
def test_rank_scope_refuses_before_read(tmp_path, rank):
    with pytest.raises(ValueError, match="rank"):
        load_witness(tmp_path, rank=rank)
    with pytest.raises(ValueError, match="rank"):
        protocol.original_names(rank)


def test_exact_generation_transport_only_own_six_files(tmp_path, originals):
    client, _, _, events = fake_cloud(1)
    root = tmp_path / "retained"
    record, witness = protocol.materialize_reference(root, rank=1, client=client)
    assert record == originals[1][0]
    assert set(witness) == set(originals[1][1])
    assert len(events) == 6
    assert sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) < protocol.MAX_REFERENCE_BYTES
    # Reattach reads already verified files; no overwrite/download.
    protocol.materialize_reference(root, rank=1, client=client)
    assert len(events) == 6
    endpoint = root / protocol.original_names(1)[-1]
    endpoint.write_bytes(b"corrupted retained evidence")
    with pytest.raises(ValueError, match="path/size/CRC"):
        protocol.materialize_reference(root, rank=1, client=client)
    assert endpoint.read_bytes() == b"corrupted retained evidence"
    assert len(events) == 6


@pytest.mark.parametrize("mutation", ["region", "crc", "size", "bytes", "disk"])
def test_transport_refuses_changed_source_and_bad_region(tmp_path, monkeypatch, mutation):
    client, bucket, blobs, events = fake_cloud(0)
    first = blobs[protocol.LEDGER_PIN["name"]]
    if mutation == "region":
        bucket.location = "EU"
    elif mutation == "crc":
        first.crc32c = "AAAAAA=="
    elif mutation == "size":
        first.size += 1
    elif mutation == "bytes":
        first.raw = bytes([first.raw[0] ^ 1]) + first.raw[1:]
    else:
        monkeypatch.setattr(protocol.shutil, "disk_usage", lambda p: NS(free=0))
    with pytest.raises((ValueError, SystemExit)):
        protocol.materialize_reference(tmp_path / "retained", rank=0, client=client)
    assert len(events) == (1 if mutation == "bytes" else 0)


def test_actual_selected_metadata_no_payload_or_backend(monkeypatch, originals):
    import jax
    import glm_tpu.greenfield.checkpoint.ws32_layer_subset as subset_module
    def forbidden(*a, **kw):
        pytest.fail("preflight must not initialize devices or read weight payload")
    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "device_put", forbidden)
    monkeypatch.setattr(subset_module, "iter_ws32_layer_subset_host_tensors", forbidden)
    pins, subset = preflight.selected_metadata(REPO, (9, 13, 25, 29))
    assert subset.layer_ids == (0, 1) and subset.include_embedding
    assert len(subset.tensor_indices) == 55
    assert subset.payload_bytes_per_chip == 102589760
    assert pins["expected_manifest_sha256"] == originals[0][0]["checkpoint_manifest_sha256"]
    names = {subset.metadata.plans[0].tensors[i].name for i in subset.tensor_indices}
    assert "model.embed_tokens.weight" in names
    assert all(n.startswith(("model.layers.0.", "model.layers.1.")) for n in names - {"model.embed_tokens.weight"})


def test_composed_original_preflight_no_runtime(tmp_path, monkeypatch, originals):
    monkeypatch.setattr(preflight, "RUN_ROOT", tmp_path)
    client, _, _, _ = fake_cloud(0)
    root = tmp_path / TAG / "rank0"
    preflight.retained_preflight(tag=TAG, rank=0, pin="a"*40, root=root, repo=REPO, client=client)
    report = json.loads((root / "retained_preflight.json").read_text())
    assert report["protocol"] == protocol.PROTOCOL and "layer" not in report
    assert report["selected_layer_ids"] == [0, 1] and report["include_embedding"]
    assert report["context_capacity"] == 8192 and report["host_main_rope_table"]
    assert {v["device_slot"] for v in report["headers"]} == {9, 13, 25, 29}
    assert report["numerical_promotion"] is False and report["performance_claim"] is False
    original = root / "retained_reference" / protocol.original_names(0)[0]
    assert report["original_runner_sha256"] == sha256(original.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        preflight.retained_preflight(tag=TAG, rank=0, pin="a"*40, root=root, repo=REPO, client=client)


@pytest.mark.parametrize("mutation", ["host", "manifest", "owner"])
def test_preflight_rejects_wrong_host_checkpoint_or_owner(tmp_path, monkeypatch, originals, mutation):
    monkeypatch.setattr(preflight, "RUN_ROOT", tmp_path)
    prior, witness = deepcopy(originals[0])
    if mutation == "host":
        prior["hostname"] = "wrong-host"
    elif mutation == "manifest":
        prior["checkpoint_manifest_sha256"] = "0"*64
    else:
        prior["local_device_slots"][0]["file_sha256"] = "0"*64
    monkeypatch.setattr(protocol, "materialize_reference", lambda *a, **kw: (prior, witness))
    root = tmp_path / TAG / "rank0"
    with pytest.raises(ValueError, match="different host|differs"):
        preflight.retained_preflight(tag=TAG, rank=0, pin="a"*40, root=root, repo=REPO, client=None)
    assert not (root / "retained_preflight.json").exists()


def test_campaign_routes_distinct_preflight_and_refuses_invalid_pin(monkeypatch):
    from scripts.greenfield import ws32_prefill_layer_campaign as campaign
    from scripts.greenfield import microbench_fp8_matmul
    from google.cloud import storage
    pin, events = "a"*40, []
    monkeypatch.setattr(microbench_fp8_matmul, "_git_head", lambda: pin)
    monkeypatch.setattr(storage, "Client", lambda: "client")
    monkeypatch.setattr(preflight, "retained_preflight", lambda **kw: events.append(kw))
    monkeypatch.setattr(campaign, "ssh", lambda *a, **kw: pytest.fail("invalid pin must refuse before SSH"))
    campaign.retained_preflight(TAG, 0, pin)
    assert events == [dict(tag=TAG, rank=0, pin=pin, root=Path("/home/gianl/glm-run") / TAG / "rank0", repo=REPO, client="client")]
    with pytest.raises(ValueError, match="worktree/pin"):
        campaign.campaign(TAG, "invalid")
    with pytest.raises(ValueError, match="rank/code"):
        campaign.retained_preflight(TAG, True, pin)
    for bad in (TAG.replace("d01", "l6"), TAG+"/../other", TAG+"_unreviewed"):
        assert not protocol.is_tag(bad)
        with pytest.raises(ValueError):
            campaign.run_root(bad)
