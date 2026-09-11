"""Small host-only originals fixtures plus the retained real metadata/oracle."""

import base64
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace as NS

import google_crc32c
import numpy as np
import pytest

from scripts.greenfield import ws32_history_preflight as preflight
from scripts.greenfield import ws32_history_protocol as protocol

REPO = Path(__file__).resolve().parents[3]
TAG = "greenfield_fp8_ws32_history_frontier_l06_20260911T190000000000000Z"


@pytest.fixture(scope="module", autouse=True)
def no_devices_or_payloads():
    import jax
    from glm_tpu.greenfield.checkpoint import ws32_layer_subset as selected
    from glm_tpu.greenfield.checkpoint import ws32_strategy_nd_dense as overlay

    def forbidden(*args, **kwargs):
        pytest.fail("preflight initialized devices, compiled, or loaded weight payloads")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("devices", "local_devices", "device_put"):
            monkeypatch.setattr(jax, name, forbidden)
        monkeypatch.setattr(jax.stages.Lowered, "compile", forbidden)
        monkeypatch.setattr(jax.stages.Compiled, "__call__", forbidden)
        monkeypatch.setattr(selected, "iter_ws32_layer_subset_host_tensors", forbidden)
        monkeypatch.setattr(overlay, "load_ws32_strategy_nd_dense_overlay", forbidden)
        yield


@pytest.fixture
def reference(monkeypatch):
    """Synthetic transport data, never numerical evidence or cloud connectivity."""
    arrays = dict(dsa_producer_layer_ids=np.asarray(protocol.PRODUCERS, np.int32),
                  dsa_selected_positions=np.zeros((1, 4, 1, 2048), np.int32),
                  dsa_selected_valid_counts=np.full((1, 4, 1), 2048, np.int32),
                  dsa_selected_scores=np.zeros((1, 4, 1, 2048), np.float32))
    owners = [dict(device_id=d, device_slot=s, expert_coordinate=s // 4,
                   feature_coordinate=s % 4, file_sha256="a" * 64)
              for d, s in zip((12, 14, 13, 15), (9, 13, 25, 29))]
    runners, pins, blobs, events = {}, {}, {}, []

    class Blob:
        def reload(self, *, if_generation_match):
            assert if_generation_match == 123
        def download_to_filename(self, name, *, if_generation_match):
            assert if_generation_match == 123
            events.append(self.name)
            Path(name).write_bytes(self.raw)

    def register(branch, form, raw):
        name = f"results/{protocol.ORIGINAL_TAGS[branch]}/host_records/runner.rank0.{form}"
        pin = dict(name=name, generation="123", size=len(raw),
                   crc32c=base64.b64encode(google_crc32c.Checksum(raw).digest()).decode(),
                   sha256=sha256(raw).hexdigest())
        blob = Blob()
        vars(blob).update(pin, raw=raw)
        blobs[name] = blob
        pins.setdefault(branch, {})[form] = pin

    def refresh(branch):
        register(branch, "json", json.dumps(runners[branch], sort_keys=True).encode())

    for branch in protocol.BRANCHES:
        capsule = BytesIO()
        np.savez(capsule, **arrays)
        register(branch, "npz", capsule.getvalue())
        runner = {key: "a" * 64 for key in preflight.CONTEXT_KEYS}
        runner.update(hostname="retained-host", jax_process_index=3, launch_process_id=0,
                      local_device_slots=deepcopy(owners), prompt_length=8155, context_capacity=8192,
                      main_rope_table={}, numerical_tensors=dict(
                          filename="runner.rank0.npz", byte_count=pins[branch]["npz"]["size"],
                          sha256=pins[branch]["npz"]["sha256"], arrays={
                              k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha256(v.tobytes()).hexdigest())
                              for k, v in arrays.items()}))
        runners[branch] = runner
        refresh(branch)

    def get_pins(repo, rank):
        assert repo == REPO and rank == 0
        return deepcopy(pins)
    monkeypatch.setattr(protocol, "original_pins", get_pins)
    def blob(name, *, generation):
        assert generation == 123
        return blobs[name]
    bucket = NS(location="US-CENTRAL2", reload=lambda: None, blob=blob)
    def get_bucket(name):
        assert name == protocol.BUCKET
        return bucket
    client = NS(bucket=get_bucket)
    return NS(runners=runners, pins=pins, blobs=blobs, events=events,
              bucket=bucket, client=client, refresh=refresh)


def test_four_generation_bound_originals_and_cached_reuse(tmp_path, reference):
    root = tmp_path / "originals"
    runners, rows, identity = preflight.materialize_originals(root, repo=REPO, rank=0, client=reference.client)
    assert runners == reference.runners
    assert len(reference.events) == 4
    assert set(p.name for p in root.iterdir()) == {f"{b}.{f}" for b in protocol.BRANCHES for f in ("json", "npz")}
    assert identity["bytes"] == sum(v["size"] for b in reference.pins.values() for v in b.values())
    assert identity["original_pins"] == reference.pins
    assert set(rows) == set(protocol.BRANCHES)
    assert all(row["positions"].shape == (4, 1, 2048) for row in rows.values())
    preflight.materialize_originals(root, repo=REPO, rank=0, client=reference.client)
    assert len(reference.events) == 4
    (root / "candidate.json").write_bytes(b"preserve corruption")
    with pytest.raises(ValueError, match="path/size/CRC"):
        preflight.materialize_originals(root, repo=REPO, rank=0, client=reference.client)
    assert (root / "candidate.json").read_bytes() == b"preserve corruption"
    assert len(reference.events) == 4


@pytest.mark.parametrize("mutation", ["region", "generation", "size", "crc", "bytes", "sha", "disk", "budget"])
def test_transport_fail_closed(tmp_path, monkeypatch, reference, mutation):
    first = reference.blobs[reference.pins["candidate"]["json"]["name"]]
    if mutation == "region":
        reference.bucket.location = "US-CENTRAL1"
    elif mutation == "generation":
        first.generation = "124"
    elif mutation == "size":
        first.size += 1
    elif mutation == "crc":
        first.crc32c = "wrong"
    elif mutation == "bytes":
        first.raw = b"x" * len(first.raw)
    elif mutation == "sha":
        reference.pins["candidate"]["json"]["sha256"] = "0" * 64
    elif mutation == "disk":
        monkeypatch.setattr(preflight.shutil, "disk_usage", lambda _: NS(free=0))
    else:
        monkeypatch.setattr(protocol, "ORIGINALS_LIMIT", 1)
    with pytest.raises((ValueError, SystemExit)):
        preflight.materialize_originals(tmp_path / "originals", repo=REPO, rank=0, client=reference.client)
    assert len(reference.events) <= (1 if mutation in ("bytes", "sha") else 0)


@pytest.mark.parametrize("mutation", ["partial", "partial_symlink", "ancestor_symlink", "file_symlink"])
def test_no_replace_or_follow_symlinks(tmp_path, reference, mutation):
    root = tmp_path / "originals"
    root.mkdir()
    target = tmp_path / "keep"
    target.write_bytes(b"keep")
    if mutation == "partial":
        (root / "candidate.json.partial").write_bytes(b"keep partial")
    elif mutation == "partial_symlink":
        (root / "candidate.json.partial").symlink_to(target)
    elif mutation == "file_symlink":
        (root / "candidate.json").symlink_to(target)
    else:
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link / "child"
    with pytest.raises((ValueError, FileExistsError)):
        preflight.materialize_originals(root, repo=REPO, rank=0, client=reference.client)
    assert reference.events == [] and target.read_bytes() == b"keep"


@pytest.mark.parametrize("mutation", ["rank", "owner", "owner_duplicate", "branch_context", "npz_sha", "array_sha", "array_shape"])
def test_original_json_npz_and_owner_bindings(tmp_path, reference, mutation):
    runner = reference.runners["control"]
    if mutation == "rank":
        runner["launch_process_id"] = 1
    elif mutation == "owner":
        runner["local_device_slots"][0]["expert_coordinate"] = 0
    elif mutation == "owner_duplicate":
        runner["local_device_slots"][0] = deepcopy(runner["local_device_slots"][1])
    elif mutation == "branch_context":
        runner["token_oracle_manifest_sha256"] = "b" * 64
    elif mutation == "npz_sha":
        runner["numerical_tensors"]["sha256"] = "b" * 64
    elif mutation == "array_sha":
        runner["numerical_tensors"]["arrays"]["dsa_selected_scores"]["sha256"] = "b" * 64
    else:
        runner["numerical_tensors"]["arrays"]["dsa_selected_scores"]["shape"] = [1, 4, 2048, 1]
    reference.refresh("control")
    with pytest.raises(ValueError):
        preflight.materialize_originals(tmp_path / "originals", repo=REPO, rank=0, client=reference.client)


@pytest.fixture(scope="module")
def actual():
    # Read retained JSON and local checkpoint/overlay HEADERS only. No copies,
    # cloud calls, model payload loading or CPU full-model compilation.
    runners = {b: json.loads((Path("/home/gianl/glm-run") / protocol.ORIGINAL_TAGS[b] / "runner.rank0.json").read_text())
               for b in protocol.BRANCHES}
    return runners


def config():
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    from glm_tpu.greenfield.types import ModelGeometry
    geometry = ModelGeometry.from_hf_config(json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text()))
    return Ws32DecoderConfig(geometry, 8192, host_main_rope_table=True)


@pytest.fixture(scope="module")
def actual_metadata(actual):
    # Authenticate the unchanged retained inventory once; mutation tests alter
    # only copies of the original runner, never this expensive shared metadata.
    slots = tuple(sorted(v["device_slot"] for v in actual["candidate"]["local_device_slots"]))
    return preflight.selected_metadata(REPO, slots)


def test_actual_retained_metadata_overlay_and_missing_prompt_hash(actual, actual_metadata):
    pins, subset = actual_metadata
    assert len(subset.tensor_indices) == 201
    assert subset.payload_bytes_per_chip == 1424692176
    overlay = preflight.bind_metadata(REPO, actual, pins, subset)
    assert len(overlay.records) == 96
    assert all("prompt_ids_sha256" not in r for r in actual.values())
    tokens, rope = preflight.host_inputs(actual, config())
    assert tokens.shape == (8155,) and sha256(tokens.tobytes()).hexdigest() == protocol.PROMPT_SHA
    assert rope.nbytes == 1048576


@pytest.mark.parametrize("mutation", ["explicit_prompt", "oracle", "length", "rope", "witness", "oracle_tokens"])
def test_host_binding_does_not_silently_accept_missing_prompt_sha(actual, monkeypatch, mutation):
    runners = deepcopy(actual)
    if mutation == "explicit_prompt":
        runners["control"]["prompt_ids_sha256"] = "b" * 64
    elif mutation == "oracle":
        runners["control"]["dsa_oracle_success_sha256"] = "b" * 64
    elif mutation == "length":
        runners["control"]["prompt_length"] -= 1
    elif mutation == "rope":
        runners["control"]["main_rope_table"]["sha256"] = "b" * 64
    else:
        from glm_tpu.greenfield.validation import ws32_short_context as oracle_module
        original = oracle_module.load_ws32_short_context_oracle
        def altered(*args, **kwargs):
            from dataclasses import replace
            oracle = original(*args, **kwargs)
            field = "generated_token_ids" if mutation == "witness" else "prompt_token_ids"
            values = getattr(oracle, field).copy()
            values[0] += 1
            return replace(oracle, **{field: values})
        monkeypatch.setattr(oracle_module, "load_ws32_short_context_oracle", altered)
    with pytest.raises(ValueError, match="history original"):
        preflight.host_inputs(runners, config())


@pytest.mark.parametrize("mutation", ["checkpoint", "owner_file", "overlay_pin", "overlay_owner"])
def test_metadata_refuses_original_ledger_drift(actual, actual_metadata, mutation):
    runners = deepcopy(actual)
    pins, subset = actual_metadata
    runner = runners["control"]
    if mutation == "checkpoint":
        runner["checkpoint_success_sha256"] = "b" * 64
    elif mutation == "owner_file":
        runner["local_device_slots"][0]["file_sha256"] = "b" * 64
    elif mutation == "overlay_pin":
        runner["strategy_nd_dense_overlay"]["manifest_file_sha256"] = "b" * 64
    else:
        runner["strategy_nd_dense_overlay"]["local_records"].pop()
    with pytest.raises(ValueError, match="history original"):
        preflight.bind_metadata(REPO, runners, pins, subset)


def test_composed_preflight_reports_no_admission(tmp_path, monkeypatch, actual, actual_metadata):
    monkeypatch.setattr(preflight, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: actual["candidate"]["hostname"])
    # Transport itself is exercised above; reuse originals without any copies.
    monkeypatch.setattr(preflight, "materialize_originals", lambda *a, **k: (actual, {}, dict(original_pins="fixture")))
    def selected(repo, slots):
        assert repo == REPO and slots == (9, 13, 25, 29)
        return actual_metadata
    monkeypatch.setattr(preflight, "selected_metadata", selected)
    root = tmp_path / TAG / "rank0"
    kwargs = dict(tag=TAG, rank=0, pin="a" * 40, root=root, repo=REPO, client=None)
    preflight.retained_preflight(**kwargs)
    report = json.loads((root / "retained_preflight.json").read_text())
    assert report["selected_layer_ids"] == list(range(7))
    assert report["selected_leaf_count"] == 201 and report["overlay_tensor_count"] == 12
    assert report["prompt_ids_sha256"] == protocol.PROMPT_SHA
    assert report["prompt_hash_source"] == "AUTHENTICATED_ORIGINAL_TOKEN_ORACLE"
    assert all(report[k] is False for k in ("numerical_execution_available", "numerical_admission", "numerical_promotion", "performance_claim"))
    assert {o["device_slot"] for o in report["local_device_slots"]} == {9, 13, 25, 29}
    with pytest.raises(FileExistsError):
        preflight.retained_preflight(**kwargs)


@pytest.mark.parametrize("mutation", ["rank", "tag", "pin", "path", "source", "metadata", "hostname"])
def test_composed_failure_never_publishes_success(tmp_path, monkeypatch, actual, mutation):
    monkeypatch.setattr(preflight, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: actual["candidate"]["hostname"])
    monkeypatch.setattr(preflight, "materialize_originals", lambda *a, **k: (actual, {}, {}))
    root = tmp_path / TAG / "rank0"
    kwargs = dict(tag=TAG, rank=0, pin="a" * 40, root=root, repo=REPO, client=None)
    if mutation == "rank":
        kwargs["rank"] = True
    elif mutation == "tag":
        kwargs["tag"] = "greenfield_fp8_ws32_dense_frontier_d01_20260911T190000000000000Z"
    elif mutation == "pin":
        kwargs["pin"] = "bad"
    elif mutation == "path":
        kwargs["root"] = tmp_path / "other"
    elif mutation == "hostname":
        monkeypatch.setattr(preflight.socket, "gethostname", lambda: "wrong-host")
    else:
        def refused(*a, **k):
            raise ValueError("fixture refusal")
        monkeypatch.setattr(preflight, "require_source" if mutation == "source" else "selected_metadata", refused)
    with pytest.raises(ValueError):
        preflight.retained_preflight(**kwargs)
    assert not (root / "retained_preflight.json").exists()
