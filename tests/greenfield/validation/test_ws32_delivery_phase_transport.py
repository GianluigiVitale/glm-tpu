"""Actual gzip/exact-generation transport against in-memory GCS; no TPU."""
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_delivery_phase_transport as transport
from glm_tpu.greenfield.validation import ws32_evidence
from tests.greenfield.validation.gcs_fixtures import Bucket

LABEL = "128k_d1_0"
TAG = "greenfield_ws32_short_decoder_128k_d1_0_numerical_c128_cap131072_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T070000000000000Z"
PIN = "a" * 40


def originals(root, rank):
    values = {}
    for name in transport.file_limits(rank):
        values[name] = (json.dumps(dict(name=name, fixture=True)) + "\n").encode()
        for graph in transport.WK_GRAPHS:
            for form in transport.FORMS:
                if name.endswith(f"/{graph}.{form}"):
                    values[name] = f"fixture graph {graph}/{form}".encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(values[name])
    return values


def publish(root, bucket, rank=0):
    return transport.publish_rank(root=root, tag=TAG, pin=PIN, label=LABEL,
        rank=rank, client=bucket.client())


def collect(destination, bucket, rank=0):
    return transport.collect_rank(destination=destination, tag=TAG, pin=PIN,
        label=LABEL, rank=rank, client=bucket.client(),
        blobs={name: bucket.get_blob(name) for name in bucket.objects})


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    root = tmp_path / TAG
    values = originals(root, 0)
    bucket = Bucket()
    manifest = publish(root, bucket)
    destination = tmp_path / "collected"
    destination.mkdir()
    return NS(root=root, values=values, bucket=bucket, manifest=manifest, destination=destination)


def test_actual_gzip_publish_original_collection_and_idempotence(case):
    assert len(case.manifest["files"]) == 48 and not case.manifest["missing"]
    assert not case.manifest["numerical_claim"]
    before = deepcopy(case.bucket.objects)
    assert publish(case.root, case.bucket) == case.manifest
    assert case.bucket.objects == before
    records = collect(case.destination, case.bucket)
    assert len(records) == 49
    for name, data in case.values.items():
        assert (case.destination / name).read_bytes() == data
        remote = transport.prefix(TAG, 0) + name + ".gz"
        packed = case.bucket.objects[remote][1]
        assert gzip.decompress(packed) == data
        row = next(row for row in records if row["name"] == remote)
        assert row["sha256"] == sha256(packed).hexdigest()
        assert row["inflated_sha256"] == sha256(data).hexdigest()
    assert collect(case.destination, case.bucket) == records
    assert {name: (case.root / name).read_bytes() for name in case.values} == case.values


@pytest.mark.parametrize("mutation", ["rank", "pin", "label", "missing", "duplicate", "traversal", "oversize", "total", "generation", "hash", "schema"])
def test_manifest_mutations_refuse_before_payload(case, mutation):
    value = deepcopy(case.manifest)
    if mutation == "rank": value["rank"] = True
    elif mutation == "pin": value["code_hash"] = "b" * 40
    elif mutation == "label": value["context_label"] = "256k_e0"
    elif mutation == "missing": value["missing"] = ["a file"]
    elif mutation == "duplicate": value["files"].append(value["files"][0])
    elif mutation == "traversal": value["files"][0]["relative_path"] = "../outside"
    elif mutation == "oversize": value["files"][0]["original_bytes"] = transport.RANK_CAP
    elif mutation == "total": value["original_bytes"] += 1
    elif mutation == "generation": value["files"][0]["generation"] = 1
    elif mutation == "hash": value["files"][0]["original_sha256"] = "A" * 64
    else: value["extra"] = True
    with pytest.raises(ValueError):
        transport.validate_manifest(value, tag=TAG, pin=PIN, label=LABEL, rank=0)


@pytest.mark.parametrize("mutation", ["generation", "payload", "bomb", "extra", "local", "link", "space", "region"])
def test_collection_refuses_damage_without_overwriting(case, monkeypatch, mutation):
    row = case.manifest["files"][0]
    if mutation in ("generation", "payload", "bomb"):
        generation, data = case.bucket.objects[row["name"]]
        if mutation == "generation": generation += 1
        else:
            data = gzip.compress(b"x" * (row["original_bytes"] + (100000 if mutation == "bomb" else 0)))
        case.bucket.objects[row["name"]] = generation, data
        if mutation == "bomb":
            # Update the compression metadata too: inflated byte/hash binding
            # must still reject, before writing the advertised local original.
            blob = case.bucket.get_blob(row["name"])
            row.update(size=blob.size, crc32c=blob.crc32c)
            name = transport.prefix(TAG, 0) + transport.MANIFEST
            generation, _ = case.bucket.objects[name]
            case.bucket.objects[name] = generation, json.dumps(case.manifest).encode()
    elif mutation == "extra":
        case.bucket.objects[transport.prefix(TAG, 0) + "extra.json"] = 123, b"extra"
    elif mutation in ("local", "link"):
        target = case.destination / row["relative_path"]
        target.parent.mkdir(parents=True)
        if mutation == "local": target.write_bytes(b"preserve old local evidence")
        else: target.symlink_to(case.root / row["relative_path"])
    elif mutation == "space": monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=1))
    else: case.bucket.location = "EU"
    with pytest.raises(ValueError): collect(case.destination, case.bucket)
    if mutation == "local": assert target.read_bytes() == b"preserve old local evidence"
    if mutation == "link": assert target.is_symlink()


def test_failure_publication_is_preserved_but_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    root = tmp_path / TAG
    root.mkdir()
    bucket = Bucket()
    value = publish(root, bucket)
    assert len(value["missing"]) == 48
    with pytest.raises(ValueError, match="incomplete"):
        transport.validate_manifest(value, tag=TAG, pin=PIN, label=LABEL, rank=0)


def test_unknown_or_oversize_original_refuses_before_any_upload(case):
    bucket = Bucket()
    path = case.root / "delivery_wk.rank0/weights.bin"
    path.write_bytes(b"not an allowed artifact")
    with pytest.raises(ValueError, match="unexpected"): publish(case.root, bucket)
    assert not bucket.objects


def test_nine_graphs_are_explicit_not_inferred_from_worker():
    assert len(ws32_evidence._graphs(exact_dsa=True)) == 7
    assert len(ws32_evidence._graphs(exact_dsa=True, delivery=True)) == 9
    with pytest.raises(ValueError): ws32_evidence._graphs(exact_dsa=False, delivery=True)
    names = ws32_evidence._hlo_object_names(exact_dsa=True,
        layout=ws32_evidence.EVIDENCE_LAYOUT_V2, delivery=True)
    assert len(names) == 18 and "hlo/wk_decode.stablehlo.mlir.gz" in names


def test_shared_wk_graph_conflict_preserves_original_generation(case):
    originals(case.root, 1)
    path = case.root / "delivery_wk.rank1/wk_decode.stablehlo.mlir"
    path.write_bytes(b"different compiler graph")
    name = f"results/{TAG}/hlo/wk_decode.stablehlo.mlir.gz"
    original = case.bucket.objects[name]
    with pytest.raises((ValueError, SystemExit, RuntimeError)):
        publish(case.root, case.bucket, 1)
    assert case.bucket.objects[name] == original
    assert transport.prefix(TAG, 1) + transport.MANIFEST not in case.bucket.objects


def test_terminal_prefix_refuses_new_publication(case):
    case.bucket.objects[f"results/{TAG}/SUCCESS"] = 9911, b"sealed"
    before = deepcopy(case.bucket.objects)
    with pytest.raises(ValueError, match="terminal SUCCESS"):
        publish(case.root, case.bucket)
    assert case.bucket.objects == before


def test_cli_wrong_host_refuses_before_idle_check(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["transport", "--tag", TAG, "--code-hash", PIN,
                                     "--context-label", LABEL, "--rank", "0"])
    monkeypatch.setattr(transport.socket, "gethostname", lambda: "worker-w-1")
    def forbidden():
        pytest.fail("wrong host reached local process guard")
    monkeypatch.setattr(transport, "require_local_idle", forbidden)
    with pytest.raises(SystemExit):
        transport.main()


@pytest.mark.parametrize("profile", [transport.PROFILE, "historical_short_profile"])
def test_actual_shell_hook_expansion_only(profile):
    # Evaluate ONLY the command-construction block, never the TPU launcher or
    # cloud uploader. This catches quoting/injection regressions without SSH.
    source = Path("scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    start = source.index("if [[ $BATCHED_PROFILE == ws32_delivery_long_phase_v1 ]]; then\n",
                         source.index("execute_command="))
    hook = source[start:source.index("\nfi\n", start) + 4]
    original = 'upload(){ local rc=0; echo original; }; trap "upload || true" EXIT;'
    script = "set -euo pipefail\n" + "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in
        dict(BATCHED_PROFILE=profile, PIN=PIN, CONTEXT=LABEL,
             execute_command=original).items()
    ) + "\n" + hook + '\nprintf "%s" "$execute_command"\n'
    result = subprocess.run(["bash", "-c", script], check=True, text=True, capture_output=True)
    if profile == transport.PROFILE:
        assert "JAX_PLATFORMS=cpu timeout" in result.stdout
        assert f"--code-hash {PIN} --context-label {LABEL}" in result.stdout
        assert '--tag "$tag"' in result.stdout and '--rank "$idx"' in result.stdout
        assert result.stdout.count("upload(){ local rc=0;") == 1
        assert 'echo original; }; trap "upload || true" EXIT;' in result.stdout
    else:
        assert result.stdout == original


def test_actual_prewrite_refuses_before_original_replacement(tmp_path, monkeypatch):
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from scripts.greenfield.ws32_compile_originals import _write_compiler_original
    from types import SimpleNamespace

    root = tmp_path / "delivery_wk.rank0"
    root.mkdir()
    path = root / "runner.json"
    _atomic_json(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="pre-write"):
        _atomic_json(path, {"overflow": "x" * (4 << 20)})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="pre-write"):
        _write_compiler_original(root, {}, "wk_decode", "stablehlo.mlir", "x" * ((8 << 20) + 1))
    assert not (root / "wk_decode.stablehlo.mlir").exists()
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda p: SimpleNamespace(free=transport.DISK_RESERVE))
    with pytest.raises(ValueError, match="disk reserve"):
        _atomic_json(path, {"small": True})
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", [None, "missing_phase", "old_mode", "wrong_profile"])
def test_actual_eight_rank_outer_materializer(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(transport, "RUN_ROOT", tmp_path / "workers")
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: NS(free=20 << 30))
    bucket = Bucket()
    def put(name, raw):
        bucket.objects[f"results/{TAG}/" + name] = 1000 + len(bucket.objects), raw
    graph_records = {}
    for graph in ws32_evidence._graphs(exact_dsa=True, delivery=True):
        graph_records[graph] = {}
        for form, key in ws32_evidence.HLO_FORMS:
            raw = f"fixture graph {graph}/{form}".encode()
            put(f"hlo/{graph}.{form}.gz", gzip.compress(raw))
            graph_records[graph][key] = sha256(raw).hexdigest()
    for rank in range(8):
        root = transport.RUN_ROOT / TAG
        originals(root, rank)
        publish(root, bucket, rank)
        trace = f"fixture trace rank{rank}".encode()
        put(f"traces/trace.rank{rank}.xplane.pb", trace)
        record = dict(code_hash=PIN, compile_only=False, launch_process_id=rank,
            status="SUCCESS", exact_dsa=True, prefill_mode="layer_major_raw_v1",
            batched_prefill_profile=transport.PROFILE, delivery_context_label=LABEL,
            evidence_layout=ws32_evidence.EVIDENCE_LAYOUT_V2, graphs=graph_records,
            trace=dict(files=[dict(sha256=sha256(trace).hexdigest(), byte_count=len(trace))]))
        if mutation == "wrong_profile" and rank == 7: record["batched_prefill_profile"] = "old"
        put(f"host_records/runner.rank{rank}.json", json.dumps(record).encode())
        put(f"host_records/runner.rank{rank}.log", b"fixture log")
        put(f"host_records/runner.rank{rank}.npz", b"fixture tensors; not numerical proof")
    if mutation == "missing_phase":
        del bucket.objects[transport.prefix(TAG, 7) + "delivery_wk.rank7/call_records/call041.json.gz"]
    client = bucket.client()
    client.list_blobs = lambda _, prefix: [bucket.get_blob(name) for name in bucket.objects if name.startswith(prefix)]
    monkeypatch.setattr(ws32_evidence.storage, "Client", lambda: client)
    destination = tmp_path / "controller"
    args = dict(run_dir=destination, remote_prefix=f"gs://{transport.BUCKET}/results/{TAG}",
        mode="numerical", tag=TAG, code_hash=PIN, recovery_code_hash=PIN,
        exact_dsa=True, allow_failure_diagnostics=True, output=destination / "source_remote_objects.json",
        delivery_context_label=None if mutation == "old_mode" else LABEL)
    if mutation is not None:
        with pytest.raises((ValueError, SystemExit)): ws32_evidence.materialize(**args)
        assert not args["output"].exists()
    else:
        result = ws32_evidence.materialize(**args)
        assert len(result["objects"]) == 442
        assert result["delivery_context_label"] == LABEL
        assert not result["failure_diagnostics_preserved"]
        assert (destination / "fleet/delivery_wk.rank7/call_records/call041.json").is_file()
        assert (destination / "fleet/delivery_decode.rank7/runner.json").is_file()
        assert len(list((destination / "fleet_hlo").glob("wk_decode.rank*.stablehlo.mlir"))) == 8
        assert all(type(row["generation"]) is int for row in result["objects"])
