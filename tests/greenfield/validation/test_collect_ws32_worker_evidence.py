"""Original bytes, topology identities and write-once remote collection."""
from copy import deepcopy
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/greenfield/collect_ws32_worker_evidence.py"
spec = importlib.util.spec_from_file_location("collect_ws32", SCRIPT)
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)
TAG, PIN = "test-tag", "a" * 40


def fixture(root, rank=0):
    root.mkdir(exist_ok=True)
    capture = dict(hostname=f"pod-w-{rank}", jax_process_index=(rank + 3) % 8)
    record = dict(code_hash=PIN, launch_process_id=rank, **capture,
                  status="SUCCESS", compile_only=False,
                  artifact_kind="greenfield_ws32_short_decoder", exact_dsa=True,
                  evidence_layout="hlo_single_gzip_v2", graphs={})
    for graph in collect.GRAPHS:
        record["graphs"][graph] = {}
        for suffix, key in collect.FORMS:
            path = root / "hlo" / f"{graph}.{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"graph {graph} {suffix}".encode())
            record["graphs"][graph][key] = collect.digest_file(path)["sha256"]
    npz = root / f"runner.rank{rank}.npz"
    npz.write_bytes(b"original NPZ payload")
    record["numerical_tensors"] = dict(filename=npz.name, byte_count=npz.stat().st_size,
                                        sha256=collect.digest_file(npz)["sha256"])
    trace = root / "trace" / "plugins" / "profile" / "trace.xplane.pb"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"original XPlane payload")
    record["trace"] = dict(files=[dict(relative_path="plugins/profile/trace.xplane.pb",
                                       byte_count=trace.stat().st_size,
                                       sha256=collect.digest_file(trace)["sha256"])])
    (root / f"runner.rank{rank}.json").write_text(json.dumps(record))
    (root / f"runner.rank{rank}.log").write_text("original completed log")
    return capture, record


def test_inventory_preserves_original_bytes_with_nontrivial_jax_mapping(tmp_path):
    capture, _ = fixture(tmp_path)
    before = (tmp_path / "runner.rank0.json").read_bytes()
    inventory = collect.original_inventory(tmp_path, TAG, PIN, 0, capture)
    assert len(inventory["files"]) == 18
    assert sum(x["compressed"] for x in inventory["files"]) == 14
    assert (tmp_path / "runner.rank0.json").read_bytes() == before
    assert not any(tmp_path.glob("**/*.gz"))


DELIVERY_TAG = "greenfield_ws32_short_decoder_128k_d0_95_numerical_c128_cap131072_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T142548940670244Z"


def delivery_fixture(root, rank=0):
    capture, record = fixture(root, rank)
    record.update(batched_prefill_profile="ws32_delivery_long_phase_v1",
                  delivery_context_label="128k_d0_95", context_capacity=131072,
                  prompt_length=127363)
    record["graphs"].update(wk_decode={}, wk_promote={})
    (root / f"runner.rank{rank}.json").write_text(json.dumps(record))
    return capture, record


def test_delivery_primary_inventory_keeps_original_record_and_separate_phase_scope(tmp_path):
    rows = []
    for rank in range(8):
        root = tmp_path / str(rank)
        capture, _ = delivery_fixture(root, rank)
        before = (root / f"runner.rank{rank}.json").read_bytes()
        with pytest.raises(ValueError):
            collect.original_inventory(root, DELIVERY_TAG, PIN, rank, capture)
        value = collect.original_inventory(root, DELIVERY_TAG, PIN, rank, capture,
                                          delivery_context_label="128k_d0_95")
        assert len(value["files"]) == 18  # phase publisher owns WK + preparation
        assert (root / f"runner.rank{rank}.json").read_bytes() == before
        rows.append(value)
    collect.require_fleet_inventories(rows, DELIVERY_TAG, PIN, delivery_context_label="128k_d0_95")
    with pytest.raises(ValueError, match="workload"):
        collect.require_fleet_inventories(rows, DELIVERY_TAG, PIN)


@pytest.mark.parametrize("mutation", ["profile", "label", "capacity", "prompt", "extra_graph", "missing_wk"])
def test_delivery_inventory_refuses_wrong_selected_workload(tmp_path, mutation):
    capture, record = delivery_fixture(tmp_path)
    if mutation == "extra_graph":
        record["graphs"]["unknown"] = {}
    elif mutation == "missing_wk":
        del record["graphs"]["wk_decode"]
    else:
        key = {"profile": "batched_prefill_profile", "label": "delivery_context_label",
               "capacity": "context_capacity", "prompt": "prompt_length"}[mutation]
        record[key] = "wrong" if mutation in ("profile", "label") else 1
    (tmp_path / "runner.rank0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        collect.original_inventory(tmp_path, DELIVERY_TAG, PIN, 0, capture,
                                   delivery_context_label="128k_d0_95")


@pytest.mark.parametrize("mutation", ["pin", "rank", "jax", "status", "trace", "npz", "hlo"])
def test_inventory_refuses_wrong_identity_and_changed_payload(tmp_path, mutation):
    capture, record = fixture(tmp_path)
    if mutation == "pin":
        record["code_hash"] = "b" * 40
    elif mutation == "rank":
        record["launch_process_id"] = 7
    elif mutation == "jax":
        record["jax_process_index"] = 0
    elif mutation == "status":
        record["status"] = "ORACLE_MISMATCH"
    elif mutation == "trace":
        record["trace"]["files"][0]["relative_path"] = "../../outside"
    elif mutation == "npz":
        (tmp_path / "runner.rank0.npz").write_bytes(b"changed")
    else:
        (tmp_path / "hlo/decode.optimized_hlo.txt").write_bytes(b"changed")
    (tmp_path / "runner.rank0.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        collect.original_inventory(tmp_path, TAG, PIN, 0, capture)


def test_shared_graphs_are_validated_across_all_ranks_before_publication(tmp_path):
    rows = []
    for rank in range(8):
        root = tmp_path / str(rank)
        capture, record = fixture(root, rank)
        rows.append(collect.original_inventory(root, TAG, PIN, rank, capture))
    collect.require_fleet_inventories(rows, TAG, PIN)
    root = tmp_path / "7"
    path = root / "hlo/decode.stablehlo.mlir"
    path.write_bytes(b"a different graph")
    record["graphs"]["decode"]["stablehlo_sha256"] = collect.digest_file(path)["sha256"]
    (root / "runner.rank7.json").write_text(json.dumps(record))
    rows[-1] = collect.original_inventory(root, TAG, PIN, 7, capture)
    with pytest.raises(ValueError, match="disagree"):
        collect.require_fleet_inventories(rows, TAG, PIN)
    with pytest.raises(ValueError, match="eight unique"):
        collect.require_fleet_inventories(rows[:-1], TAG, PIN)


class Blob:
    def __init__(self, bucket, name, generation=None):
        self.bucket, self.name, self.generation = bucket, name, generation

    @property
    def size(self):
        return len(self.bucket.objects[self.name])

    @property
    def crc32c(self):
        crc = collect.google_crc32c.Checksum(self.bucket.objects[self.name])
        return collect.base64.b64encode(crc.digest()).decode()

    def upload_from_file(self, stream, **kwargs):
        assert kwargs == dict(if_generation_match=0, checksum="crc32c")
        assert self.name not in self.bucket.objects
        self.bucket.objects[self.name] = stream.read()
        self.bucket.uploads += 1

    def download_to_file(self, stream, **kwargs):
        assert kwargs == dict(if_generation_match=42, checksum="crc32c")
        assert self.generation == 42
        self.bucket.reads += 1
        stream.write(self.bucket.objects[self.name])


class Bucket:
    def __init__(self):
        self.objects, self.uploads, self.reads = {}, 0, 0

    def blob(self, name, generation=None):
        return Blob(self, name, generation)

    def get_blob(self, name):
        return self.blob(name, 42) if name in self.objects else None


@pytest.mark.parametrize("compressed", [False, True])
def test_conditional_upload_and_idempotent_pinned_readback(tmp_path, compressed):
    source = tmp_path / "original"
    source.write_bytes(b"original bytes")
    expected = collect.digest_file(source)
    bucket = Bucket()
    first = collect.publish_exact(bucket, "object", source, expected, compressed=compressed)
    second = collect.publish_exact(bucket, "object", source, expected, compressed=compressed)
    assert first == second
    assert bucket.uploads == 1 and bucket.reads == 2
    assert first["generation"] == "42"


def test_accept_original_gzip_container_without_replacing_it(tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"raw graph")
    bucket = Bucket()
    original_container = gzip.compress(source.read_bytes(), compresslevel=1, mtime=123)
    bucket.objects["object"] = original_container
    collect.publish_exact(bucket, "object", source, collect.digest_file(source), compressed=True)
    assert bucket.uploads == 0 and bucket.objects["object"] == original_container


@pytest.mark.parametrize("compressed", [False, True])
def test_refuse_existing_remote_payload_without_overwrite(tmp_path, compressed):
    source = tmp_path / "original"
    source.write_bytes(b"original")
    bucket = Bucket()
    wrong = gzip.compress(b"different") if compressed else b"differnt"
    bucket.objects["object"] = wrong
    with pytest.raises(ValueError, match="differs|exceeds"):
        collect.publish_exact(bucket, "object", source, collect.digest_file(source), compressed=compressed)
    assert bucket.uploads == 0 and bucket.objects["object"] == wrong
