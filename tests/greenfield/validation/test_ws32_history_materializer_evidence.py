"""Independent bounded fleet replay from saved capsules and selected CPU bytes."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_history_materializers as capture
from scripts.greenfield import ws32_history_materializer_evidence as evidence
from scripts.greenfield import ws32_history_protocol as protocol

REPO = Path(__file__).resolve().parents[3]


def geometry_config():
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    from glm_tpu.greenfield.types import ModelGeometry
    geometry = ModelGeometry.from_hf_config(json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text()))
    return Ws32DecoderConfig(geometry, 8192, exact_dsa=True, host_main_rope_table=True)


def raw_globals():
    result = {}
    for field, shape, dtype, _ in capture.RAW:
        rows = np.arange(shape[0], dtype=np.int32)[:, None]
        columns = np.arange(shape[1], dtype=np.int32)[None, :]
        if dtype == "uint8":
            value = ((rows // 32 + columns // 128) % 16 + 0x28).astype(np.uint8)
            if field == "wk_bits_local":
                value[0, 0] = 0  # Promotion's +0 -> -0 mutation is numerically equal.
        elif dtype == "float32":
            value = (((rows + columns) % 3 + 1) * 0.125).astype(np.float32)
        else:
            value = ((rows - columns // 256) * 0.0625).astype(ml_dtypes.bfloat16)
            value[0, 0] = np.asarray(0x8000, np.uint16).view(ml_dtypes.bfloat16)
        result[field] = value
    return result


def decoded_globals(raw):
    q = raw["q_a_bits_local"].reshape(32, 64, 6144).transpose(0, 2, 1)
    kv = raw["kv_a_bits_local"].reshape(32, 18, 6144).transpose(0, 2, 1)
    q_scale = np.repeat(raw["q_a_scale_local"], 128, axis=0)[:2048].reshape(32, 64, 48).transpose(0, 2, 1)
    kv_scale = np.repeat(raw["kv_a_scale_local"], 128, axis=0)[:576].reshape(32, 18, 48).transpose(0, 2, 1)
    def dequant(bits, scales):
        return bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
    return dict(qkv_a_bits=np.concatenate((q, kv), axis=-1),
        qkv_a_scale=np.concatenate((q_scale, kv_scale), axis=-1),
        wq_b_weight_local=dequant(raw["wq_b_bits_local"], raw["wq_b_scale_local"]),
        wk_weight_bf16=dequant(raw["wk_bits_local"], raw["wk_scale_local"]).astype(ml_dtypes.bfloat16),
        head_weight_local=raw["head_weight_local"])


def source_value(raw, slot, name):
    field = next(field for field, *_ in capture.RAW if name in
                 [capture.source_names(layer)[field] for layer in protocol.PRODUCERS])
    spec = next(spec for spec in capture.RAW if spec[0] == field)
    return raw[field][capture._index(spec[1], spec[3], slot)]


def bindings_for(raw):
    owners = {slot: dict(full_file_sha256=sha256(f"file{slot}".encode()).hexdigest(), selected={
        name: evidence.digest(source_value(raw, slot, name)) for layer in protocol.PRODUCERS
        for name in capture.source_names(layer).values()}) for slot in range(32)}
    overlay = NS(manifest={"manifest_sha256": "b" * 64}, manifest_file_sha256="c" * 64,
        success_file_sha256="d" * 64, records={(layer, expert, feature): {
            "sha256": sha256(f"overlay{layer}/{expert}/{feature}".encode()).hexdigest()}
            for layer in range(3) for expert in range(8) for feature in range(4)})
    context = {key: sha256(key.encode()).hexdigest() for key in ("checkpoint_manifest_sha256",
        "checkpoint_success_sha256", "source_inventory_sha256", "mesh_sha256", "topology_sha256")}
    return evidence.CheckpointBindings(geometry_config(), context, owners, overlay)


def owner_maps():
    first = {12: 9, 14: 13, 13: 25, 15: 29}
    rest_slots = sorted(set(range(32)) - set(first.values()))
    rest_devices = sorted(set(range(32)) - set(first), reverse=True)
    return [first, *[dict(zip(rest_devices[i:i + 4], rest_slots[i:i + 4], strict=True)) for i in range(0, 28, 4)]]


def runtime_record(rank, slots, bindings):
    overlay = bindings.overlay
    return dict(launch_rank=rank, jax_process_index=(rank + 3) % 8, **bindings.context,
        local_device_slots=[dict(device_id=device, device_slot=slot,
            expected_full_file_sha256_not_verified=bindings.owners[slot]["full_file_sha256"],
            observed_selected_tensor_sha256=bindings.owners[slot]["selected"], selected_payload_bytes=protocol.PAYLOAD_BYTES)
            for device, slot in slots.items()],
        strategy_nd_dense_overlay=dict(manifest_sha256=overlay.manifest["manifest_sha256"],
            manifest_file_sha256=overlay.manifest_file_sha256, success_file_sha256=overlay.success_file_sha256,
            local_records=[dict(device_id=device, expert_coordinate=slot // 4, feature_coordinate=slot % 4,
                layer_id=layer, file_sha256=overlay.records[layer, slot // 4, slot % 4]["sha256"])
                for device, slot in slots.items() for layer in range(3)]), materializer_originals={})


def rows(spec, slots, values, *, extra=None, hash_cache):
    result = []
    for device, slot in slots.items():
        index = capture._index(spec[1], spec[3], slot)
        key = (id(values), tuple(part.indices(size) for part, size in zip(index, spec[1], strict=True)))
        if key not in hash_cache:
            hash_cache[key] = evidence.digest(values[index])
        result.append(dict(evidence._expected_row(spec, slot, device), sha256=hash_cache[key],
                           **({} if extra is None else extra(slot))))
    return result


def save_rank(root, rank, slots, bindings, raw, decoded, hash_cache):
    root.mkdir()
    (root / "materializers").mkdir()
    record = runtime_record(rank, slots, bindings)
    wk_fp32 = decoded["wk_weight_bf16"].astype(np.float32)
    for layer in protocol.PRODUCERS:
        for name in ("wk_decode", "wk_promote"):
            key = f"layer{layer}/{name}"
            report = capture._base(key, slots, record)
            report["valid"] = True
            arrays = {}
            specs = (("result", *capture.DECODED[3][1:]), ("input0", *capture.RAW[6][1:]), ("input1", *capture.RAW[7][1:])) if name == "wk_decode" else (
                ("result", (128, 6144), "float32", (None, None)), ("input0", *capture.DECODED[3][1:]))
            globals_ = (decoded["wk_weight_bf16"], raw["wk_bits_local"], raw["wk_scale_local"]) if name == "wk_decode" else (wk_fp32, decoded["wk_weight_bf16"])
            for spec, value in zip(specs, globals_, strict=True):
                role = spec[0]
                extra = (lambda slot, role=role: dict(checkpoint_source=evidence._source(bindings, layer,
                    "wk_bits_local" if role == "input0" else "wk_scale_local", slot))) if name == "wk_decode" and role != "result" else None
                report["leaves"][role] = rows(spec, slots, value, extra=extra, hash_cache=hash_cache)
                for slot in slots.values():
                    local = value[capture._index(spec[1], spec[3], slot)]
                    arrays[f"slot{slot}_{role}"] = local.view(np.uint16) if str(local.dtype) == "bfloat16" else local
            path = root / "materializers" / capture._FILENAMES[key]
            # Independent original-reader fixture: valid compressed NPZ keeps
            # eight rank fixtures small while expanded payload checks stay real.
            np.savez_compressed(path, **arrays)
            report["original"] = capture._descriptor(path, raw_bytes=sum(value.nbytes for value in arrays.values()))
            record["materializer_originals"][key] = report
    for name, inputs, outputs in (("exact_decode", capture.RAW, capture.DECODED),
                                  ("exact_promote", capture.DECODED, capture.PROMOTED)):
        report = capture._base(name, slots, record)
        report.update(valid=True, full_exact_output_originals_preserved=False,
                      exact_decode_reconstruction_checked=False, source_leaves={})
        for category, specs in (("source_leaves", inputs), ("leaves", outputs)):
            for layer in protocol.PRODUCERS:
                for spec in specs:
                    field = spec[0]
                    context = f"layer{layer}/{field}"
                    actual_field = "wq_b_weight_local" if field.startswith("wq_b_weight_aliases/") else field
                    value = raw[field] if name == "exact_decode" and category == "source_leaves" else (
                        wk_fp32 if field == "wk_weight" else decoded[actual_field])
                    def extra(slot, layer=layer, field=field, actual_field=actual_field):
                        result = {}
                        if category == "source_leaves":
                            if name == "exact_decode":
                                result["checkpoint_source"] = evidence._source(bindings, layer, field, slot)
                            else:
                                result["exact_decode_original_sha256"] = evidence.digest(decoded[field][capture._index(spec[1], spec[3], slot)])
                        else:
                            if field in ("wk_weight", "wk_weight_bf16"):
                                result["wk_original_sha256"] = evidence.digest(wk_fp32 if field == "wk_weight" else decoded["wk_weight_bf16"])
                            if name == "exact_promote":
                                result["promotion_input_sha256"] = evidence.digest(value[capture._index(spec[1], spec[3], slot)])
                        return result
                    report[category][context] = rows(spec, slots, value, extra=extra, hash_cache=hash_cache)
        original = capture._json(root, name, report)
        record["materializer_originals"][name] = dict(schema=capture.SCHEMA, name=name, completed=True,
            valid=True, errors=[], original=original)
    return record


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    root = tmp_path_factory.mktemp("history-materializer-replay")
    raw = raw_globals()
    decoded = decoded_globals(raw)
    bindings = bindings_for(raw)
    maps, records, reports, cache = owner_maps(), [], [], {}
    for rank, slots in enumerate(maps):
        path = root / f"rank{rank}"
        record = save_rank(path, rank, slots, bindings, raw, decoded, cache)
        records.append(record)
        reports.append(evidence.replay_rank(path, record, slots, bindings))
    yield NS(root=root, raw=raw, decoded=decoded, bindings=bindings, maps=maps, records=records, reports=reports)
    # Only this fixture-created private directory, never repository originals.
    shutil.rmtree(root)


def test_complete_fleet_reconstruction_reads_exactly_unique_selected_tiles(fleet):
    calls = []
    def read(slot, name):
        value = source_value(fleet.raw, slot, name)
        calls.append((slot, name, value.nbytes))
        assert ".mlp." not in name and ".indexer.wk." not in name
        return value
    report = evidence.replay_fleet(fleet.reports, bindings=fleet.bindings, read_source=read)
    assert len(calls) == len({(slot, name) for slot, name, _ in calls}) == 256
    assert sum(size for _, _, size in calls) == evidence.SOURCE_READ_BYTES == 99_639_040
    assert max(size for _, _, size in calls) == 3_145_728
    assert report["exact_decode_reconstruction_checked"] and report["exact_promote_reconstruction_checked"]
    assert report["physical_owners"] == 32 and report["rank_count"] == 8
    assert report["wk_originals_replayed"] == 64 and report["exact_receipts_replayed"] == 16
    assert not report["numerical_promotion"] and not report["history_or_original_event_reproduction"]
    assert not report["runtime_hbm_admission"] and report["overlay_payload_bytes_read"] == 0
    assert sum(path.stat().st_size for path in fleet.root.rglob("*.npz")) < 8 << 20


@pytest.fixture()
def rank_copy(tmp_path, fleet):
    root = tmp_path / "rank0"
    shutil.copytree(fleet.root / "rank0", root)
    return NS(root=root, record=deepcopy(fleet.records[0]), slots=fleet.maps[0], bindings=fleet.bindings)


def rewrite_receipt(state, name, mutate):
    path = state.root / "materializers" / f"{name}.json"
    report = json.loads(path.read_text())
    mutate(report)
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    original = state.record["materializer_originals"][name]["original"]
    original.update(bytes=len(raw), sha256=sha256(raw).hexdigest())


@pytest.mark.parametrize("defect", ["selected_sha", "selected_file", "overlay", "process_bool", "context", "slots", "missing", "extra", "npz_sha", "npz_path", "refused", "json_sha", "json_scope"])
def test_rank_original_identity_refusals(rank_copy, defect):
    state = rank_copy
    if defect == "selected_sha":
        name = next(iter(state.record["local_device_slots"][0]["observed_selected_tensor_sha256"]))
        state.record["local_device_slots"][0]["observed_selected_tensor_sha256"][name] = "0" * 64
    elif defect == "selected_file":
        state.record["local_device_slots"][0]["expected_full_file_sha256_not_verified"] = "0" * 64
    elif defect == "overlay":
        state.record["strategy_nd_dense_overlay"]["local_records"][0]["file_sha256"] = "0" * 64
    elif defect == "process_bool":
        state.record["jax_process_index"] = True
    elif defect == "context":
        state.record["checkpoint_manifest_sha256"] = "0" * 64
    elif defect == "slots":
        state.slots = {12: 8, 14: 13, 13: 25, 15: 29}
    elif defect == "missing":
        state.record["materializer_originals"].pop("layer6/wk_promote")
    elif defect == "extra":
        (state.root / "materializers/unexpected").touch()
    elif defect == "npz_sha":
        state.record["materializer_originals"]["layer0/wk_decode"]["original"]["sha256"] = "0" * 64
    elif defect == "npz_path":
        state.record["materializer_originals"]["layer0/wk_decode"]["original"]["path"] = "../other.npz"
    elif defect == "refused":
        state.record["materializer_originals"]["layer0/wk_decode"]["valid"] = False
    elif defect == "json_sha":
        state.record["materializer_originals"]["exact_decode"]["original"]["sha256"] = "0" * 64
    else:
        rewrite_receipt(state, "exact_decode", lambda report: report.update(exact_decode_reconstruction_checked=True))
    with pytest.raises(ValueError):
        evidence.replay_rank(state.root, state.record, state.slots, state.bindings)


@pytest.mark.parametrize("defect", ["owner", "slice", "shape", "dtype", "finite", "duplicate", "missing", "source_sha", "source_name", "query_alias", "wk_link", "decode_input"])
def test_rank_exact_rows_are_rederived_not_trusted(rank_copy, defect):
    state = rank_copy
    name = "exact_promote" if defect in ("query_alias", "decode_input") else "exact_decode"
    def mutate(report):
        if defect in ("source_sha", "source_name"):
            row = report["source_leaves"]["layer2/q_a_bits_local"][0]
        elif defect == "query_alias":
            row = report["leaves"]["layer2/wq_b_weight_aliases/3"][0]
        elif defect == "decode_input":
            row = report["source_leaves"]["layer2/wq_b_weight_local"][0]
        elif defect == "wk_link":
            row = report["leaves"]["layer2/wk_weight_bf16"][0]
        else:
            row = report["leaves"]["layer2/qkv_a_bits"][0]
        if defect == "owner":
            row["device_id"] = 31
        elif defect == "slice":
            row["index"][0][0] = 1
        elif defect == "shape":
            row["shape"][0] -= 1
        elif defect == "dtype":
            row["dtype"] = "float32"
        elif defect == "finite":
            row["finite"] = False
        elif defect == "duplicate":
            report["leaves"]["layer2/qkv_a_bits"][1] = deepcopy(row)
        elif defect == "missing":
            report["leaves"].pop("layer6/qkv_a_bits")
        elif defect == "source_name":
            row["checkpoint_source"]["tensor"] = "model.layers.2.mlp.strategy_nd.fake"
        elif defect == "wk_link":
            row["wk_original_sha256"] = "0" * 64
        else:
            row["sha256"] = "0" * 64
    rewrite_receipt(state, name, mutate)
    with pytest.raises(ValueError):
        evidence.replay_rank(state.root, state.record, state.slots, state.bindings)


@pytest.mark.parametrize("defect", ["nonfinite", "source", "signed_zero", "shape", "extra"])
def test_rank_actual_wk_arrays_are_reconstructed(rank_copy, defect):
    state = rank_copy
    key = "layer0/wk_promote" if defect in ("nonfinite", "signed_zero") else "layer0/wk_decode"
    report = state.record["materializer_originals"][key]
    path = state.root / report["original"]["path"]
    with np.load(path, allow_pickle=False) as loaded:
        arrays = dict(loaded)
    role = "result" if key.endswith("promote") else "input0"
    name = f"slot9_{role}"
    if defect == "nonfinite":
        arrays[name].flat[0] = np.nan
    elif defect == "source":
        arrays[name].flat[0] += 1
    elif defect == "signed_zero":
        arrays[name].view(np.uint32).flat[0] = 0x80000000
    elif defect == "shape":
        arrays[name] = arrays[name][1:]
    else:
        arrays["extra"] = np.zeros(1, np.uint8)
    np.savez_compressed(path, **arrays)
    report["original"] = capture._descriptor(path, raw_bytes=sum(value.nbytes for value in arrays.values()))
    # Rebind the row SHA as well: the source/promotion equations must still fail.
    if defect in ("source", "signed_zero", "nonfinite"):
        next(row for row in report["leaves"][role] if row["slot"] == 9)["sha256"] = evidence.digest(arrays[name])
    with pytest.raises(ValueError):
        evidence.replay_rank(state.root, state.record, state.slots, state.bindings)


@pytest.mark.parametrize("defect", ["rank", "process", "rank_bool", "device", "slot", "leaf_owner", "replica", "source", "promoted", "wk"])
def test_fleet_join_refuses_before_any_source_read(fleet, defect):
    reports = deepcopy(fleet.reports)
    first = reports[0]
    if defect == "rank":
        reports[0] = replace(first, rank=1)
    elif defect == "process":
        reports[0] = replace(first, process_index=4)
    elif defect == "rank_bool":
        reports[0] = replace(first, rank=False)
    elif defect == "device":
        slots = dict(first.local_slots)
        slot = slots.pop(12)
        slots[31] = slot
        reports[0] = replace(first, local_slots=slots)
    elif defect == "slot":
        slots = dict(first.local_slots)
        slots[12] = 0
        reports[0] = replace(first, local_slots=slots)
    elif defect == "leaf_owner":
        value = first.decoded["layer0/qkv_a_bits"].pop(9)
        first.decoded["layer0/qkv_a_bits"][0] = value
    elif defect == "replica":
        first.decoded["layer0/qkv_a_bits"][9] = "0" * 64
    elif defect in ("source", "promoted"):
        attribute = "sources" if defect == "source" else "promoted"
        context = "layer0/q_a_bits_local" if defect == "source" else "layer0/wq_b_weight_aliases/3"
        for report in reports:
            for slot in getattr(report, attribute)[context]:
                getattr(report, attribute)[context][slot] = "0" * 64
    else:
        first.wk[0][9] = ("0" * 64, "0" * 64)
    with pytest.raises(ValueError):
        evidence.replay_fleet(reports, bindings=fleet.bindings, read_source=lambda *args: pytest.fail("read before fleet admission"))


@pytest.mark.parametrize("defect", ["missing_reader", "dtype", "shape", "sha", "budget", "not_cpu", "assertion_only"])
def test_source_reader_is_mandatory_bounded_and_rechecked(fleet, defect, monkeypatch):
    calls = []
    def read(slot, name):
        value = source_value(fleet.raw, slot, name)
        calls.append((slot, name))
        if defect == "dtype":
            return value.astype(np.float32)
        if defect == "shape":
            return value[1:]
        if defect == "sha":
            value = value.copy()
            value.flat[0] ^= 1
        if defect == "assertion_only":
            return {"passed": True, "sha256": evidence.digest(value)}
        return value
    if defect == "budget":
        monkeypatch.setattr(evidence, "SOURCE_READ_LIMIT", 1)
    if defect == "not_cpu":
        monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    with pytest.raises(ValueError):
        evidence.replay_fleet(fleet.reports, bindings=fleet.bindings, read_source=None if defect == "missing_reader" else read)
    assert len(calls) == (0 if defect in ("missing_reader", "budget", "not_cpu") else 1)


def test_every_exact_output_hash_is_reconstructed_even_if_all_workers_agree(fleet):
    reports = deepcopy(fleet.reports)
    for report in reports:
        for slot in report.local_slots.values():
            report.decoded["layer0/qkv_a_bits"][slot] = "0" * 64
            report.promoted["layer0/qkv_a_bits"][slot] = "0" * 64
    calls = []
    def read(slot, name):
        calls.append((slot, name))
        return source_value(fleet.raw, slot, name)
    with pytest.raises(ValueError, match="decoded reconstruction differs"):
        evidence.replay_fleet(reports, bindings=fleet.bindings, read_source=read)
    assert len(calls) == 16  # Only the first producer's four feature-sharded q/kv leaves.


def test_selected_fp8_source_finiteness_is_checked_after_ledger_sha(fleet):
    reports = deepcopy(fleet.reports)
    bindings = deepcopy(fleet.bindings)
    changed = fleet.raw["q_a_bits_local"].copy()
    changed[0, 0] = 0x7F
    tensor = capture.source_names(0)["q_a_bits_local"]
    for slot in range(0, 32, 4):
        value = evidence.digest(changed[:, :1536])
        bindings.owners[slot]["selected"][tensor] = value
        for report in reports:
            if slot in report.sources["layer0/q_a_bits_local"]:
                report.sources["layer0/q_a_bits_local"][slot] = value
    def read(slot, name):
        return changed[:, :1536] if (slot, name) == (0, tensor) else source_value(fleet.raw, slot, name)
    with pytest.raises(ValueError, match="non-finite bits"):
        evidence.replay_fleet(reports, bindings=bindings, read_source=read)


def test_original_category_byte_ceiling_is_checked_before_opening_npz(rank_copy, monkeypatch):
    path = rank_copy.root / "materializers/layer0_wk_decode.npz"
    with path.open("r+b") as stream:
        stream.truncate(capture.WK_ORIGINALS_LIMIT)
    monkeypatch.setattr(evidence, "read_npz", lambda *args, **kwargs: pytest.fail("read before aggregate cap"))
    with pytest.raises(ValueError, match="aggregate budget"):
        evidence.replay_rank(rank_copy.root, rank_copy.record, rank_copy.slots, rank_copy.bindings)


def test_actual_metadata_factory_opens_no_selected_or_overlay_payload(monkeypatch):
    import jax
    opened = []
    original_open = Path.open
    def checked(path, *args, **kwargs):
        assert path.suffix not in (".safetensors", ".bin"), path
        if "/glm-ws32-runtime/" in str(path):
            assert path.name in ("manifest.json", "SUCCESS"), path
        opened.append(path)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked)
    monkeypatch.setattr(jax, "device_put", lambda *a, **k: pytest.fail("metadata concrete placement"))
    monkeypatch.setattr(jax.stages.Lowered, "compile", lambda *a, **k: pytest.fail("metadata compilation"))
    monkeypatch.setattr(evidence, "_cpu_functions", lambda: pytest.fail("metadata reconstruction"))
    bindings = evidence.checkpoint_bindings(REPO)
    assert set(bindings.owners) == set(range(32))
    assert all(len(owner["selected"]) == 201 for owner in bindings.owners.values())
    assert set(bindings.context) == {"checkpoint_manifest_sha256", "checkpoint_success_sha256",
        "source_inventory_sha256", "mesh_sha256", "topology_sha256"}
    for slot, owner in bindings.owners.items():
        assert set(owner["header"]) == {"device_slot", "filename", "file_bytes", "header_sha256"}
        assert owner["header"]["device_slot"] == slot
        assert owner["header"]["filename"] == f"device_slot_{slot:02d}.safetensors"
        assert owner["header"]["file_bytes"] > 1 << 30
        assert capture._HEX.fullmatch(owner["header"]["header_sha256"])
    assert len(bindings.overlay.records) == 96
    assert any(path.name == "manifest.json" for path in opened)
