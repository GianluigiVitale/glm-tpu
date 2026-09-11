"""Bounded materializer lifecycle/negative checks, no global gather or TPU."""

from collections import namedtuple
from hashlib import sha256
import json
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_history_materializers as capture
from scripts.greenfield import ws32_history_protocol as protocol


# The real launch-rank-zero physical map: four expert coordinates, feature one.
SLOTS = {12: 9, 14: 13, 13: 25, 15: 29}


class GlobalArray:
    is_fully_addressable = False

    def __init__(self, shape, dtype, shards):
        self.shape, self.dtype, self.addressable_shards = shape, np.dtype(dtype), shards

    def __array__(self, *args, **kwargs):
        raise AssertionError("attempted global array coercion")


def array(spec, slots=SLOTS, *, platform="cpu", process=3):
    field, shape, dtype, axes = spec
    shards = []
    for device, slot in slots.items():
        index = capture._index(shape, axes, slot)
        local_shape = tuple(len(range(*part.indices(size))) for part, size in zip(index, shape, strict=True))
        # Production geometry, lazily represented host payload; capture must copy
        # just the local leaf. No 4-producer global tree is allocated by fixtures.
        fill = 1 if "scale" in field else 0
        data = np.broadcast_to(np.asarray(fill, dtype=dtype), local_shape)
        shards.append(NS(device=NS(id=device, process_index=process, platform=platform), index=index, data=data))
    return GlobalArray(shape, dtype, shards)


def tree(specs, slots=SLOTS):
    fields = tuple(dict.fromkeys(spec[0].split("/")[0] for spec in specs))
    layer_type = namedtuple("Layer", fields)
    layers = []
    for _ in protocol.PRODUCERS:
        kwargs = {}
        for spec in specs:
            parts = spec[0].split("/")
            if len(parts) == 1:
                kwargs[parts[0]] = array(spec, slots)
            else:
                kwargs.setdefault(parts[0], []).append(array(spec, slots))
        layers.append(layer_type(**{key: tuple(value) if isinstance(value, list) else value for key, value in kwargs.items()}))
    return tuple(layers)


def record(raw, slots=SLOTS):
    rows = []
    for device, slot in slots.items():
        selected = {}
        for layer, values in zip(protocol.PRODUCERS, raw, strict=True):
            for field in values._fields:
                shard = next(v for v in getattr(values, field).addressable_shards if v.device.id == device)
                selected[capture.source_names(layer)[field]] = sha256(shard.data.tobytes()).hexdigest()
        rows.append(dict(device_id=device, device_slot=slot,
            expected_full_file_sha256_not_verified="a" * 64,
            observed_selected_tensor_sha256=selected))
    return dict(jax_process_index=3, local_device_slots=rows)


@pytest.fixture()
def state(tmp_path):
    # One real production slice keeps negative sweeps bounded; the full lifecycle
    # below separately covers every physical owner and all four producers.
    slots = {12: 9}
    raw = tree(capture.RAW, slots)
    decoded = tree(capture.DECODED, slots)
    promoted = tree(capture.PROMOTED, slots)
    return NS(root=tmp_path, bound=NS(raw_exact=raw, local_slots=slots),
              record=record(raw, slots), raw=raw, decoded=decoded, promoted=promoted)


def wk_calls(state):
    for layer, raw, decoded, promoted in zip(protocol.PRODUCERS, state.raw, state.decoded, state.promoted, strict=True):
        for name, value, inputs in (("wk_decode", decoded.wk_weight_bf16, (raw.wk_bits_local, raw.wk_scale_local)),
                                    ("wk_promote", promoted.wk_weight, (decoded.wk_weight_bf16,))):
            capture.capture_wk(state.root, state.record, state.bound.local_slots,
                layer=layer, name=name, value=value, inputs=inputs, platform="cpu")


def exact(state, name="exact_decode"):
    capture.capture_exact(state.root, state.record, state.bound, name=name,
        value=state.decoded if name == "exact_decode" else state.promoted,
        inputs=(state.raw if name == "exact_decode" else state.decoded,), platform="cpu")


def edit(value, flat_index, new, owner=0, *, bits=False):
    shard = value.addressable_shards[owner]
    shard.data = np.array(shard.data, copy=True)
    target = shard.data.view(np.uint16 if shard.data.dtype == ml_dtypes.bfloat16 else np.uint32) if bits else shard.data
    target.flat[flat_index] = new


def test_four_producer_four_owner_lifecycle_and_strict_storage(tmp_path):
    raw, decoded, promoted = tree(capture.RAW), tree(capture.DECODED), tree(capture.PROMOTED)
    state = NS(root=tmp_path, bound=NS(raw_exact=raw, local_slots=SLOTS),
               record=record(raw), raw=raw, decoded=decoded, promoted=promoted)
    wk_calls(state)
    exact(state)
    exact(state, "exact_promote")
    originals = state.record["materializer_originals"]
    assert set(originals) == {*capture._WK_KEYS, "exact_decode", "exact_promote"}
    wk = [originals[key] for key in capture._WK_KEYS]
    assert sum(row["original"]["raw_array_bytes"] for row in wk) == capture.WK_RAW_BYTES_FOUR_OWNERS
    assert sum(row["original"]["bytes"] for row in wk) < capture.WK_ORIGINALS_LIMIT
    paths = list((tmp_path / "materializers").iterdir())
    assert len(paths) == 10 and not any("pending" in p.name for p in paths)
    assert sum(p.stat().st_size for p in paths) < capture.TOTAL_LIMIT == 147 << 20
    for name, leaf_count in (("exact_decode", 20), ("exact_promote", 32)):
        report = capture.load_exact_receipt(tmp_path, state.record, name)
        assert report["valid"] and not report["errors"]
        assert report["local_slots"] == {str(k): v for k, v in SLOTS.items()}
        assert len(report["leaves"]) == leaf_count
        assert not report["numerical_admission"] and not report["legacy_bit_identity"]
        assert not report["full_exact_output_originals_preserved"]
        assert not report["exact_decode_reconstruction_checked"]
        for rows in report["leaves"].values():
            assert len(rows) == 4 and len({row["sha256"] for row in rows}) == 1
            assert {row["slot"] for row in rows} == set(SLOTS.values())
        query = report["leaves"]["layer6/" + ("wq_b_weight_local" if name == "exact_decode" else "wq_b_weight_aliases/3")]
        assert all(row["index"] == [[1024, 2048, 1], [0, 2048, 1]] for row in query)
    source = capture.load_exact_receipt(tmp_path, state.record, "exact_decode")["source_leaves"]
    assert all("checkpoint_source" in row for rows in source.values() for row in rows)


@pytest.mark.parametrize("defect", ["owner", "process", "platform", "slice", "local_shape", "global_shape", "dtype", "missing", "duplicate", "producer_tree", "schema", "binding", "source_tree"])
def test_exact_structural_refusals_precede_reads_or_files(state, defect, monkeypatch):
    value = state.decoded[0].qkv_a_bits
    shard = value.addressable_shards[0]
    if defect == "owner":
        shard.device.id = 88
    elif defect == "process":
        shard.device.process_index = 0
    elif defect == "platform":
        shard.device.platform = "tpu"
    elif defect == "slice":
        shard.index = (slice(1, 32), slice(None), slice(None))
    elif defect == "local_shape":
        shard.data = np.zeros((1,), np.uint8)
    elif defect == "global_shape":
        value.shape = (1,)
    elif defect == "dtype":
        value.dtype = np.dtype(np.float32)
    elif defect == "missing":
        value.addressable_shards.clear()
    elif defect == "duplicate":
        value.addressable_shards *= 2
    elif defect == "producer_tree":
        state.decoded = state.decoded[:3]
    elif defect == "schema":
        state.decoded = (tuple(state.decoded[0]), *state.decoded[1:])
    elif defect == "binding":
        state.record["local_device_slots"][0]["observed_selected_tensor_sha256"].clear()
    else:
        state.raw = tuple(list(state.raw))
    monkeypatch.setattr(capture.owner_capture, "_read", lambda *a: pytest.fail("structural refusal read payload"))
    with pytest.raises(ValueError):
        exact(state)
    assert not list(state.root.iterdir())


@pytest.mark.parametrize("defect", ["nonfinite", "checkpoint", "wk_link", "signed_zero"])
def test_wk_completed_originals_precede_numerical_refusal(state, defect):
    raw = state.raw[0]
    output = state.decoded[0].wk_weight_bf16
    if defect == "nonfinite":
        edit(output, 3, 0x7FC3, bits=True)
    elif defect == "checkpoint":
        edit(raw.wk_bits_local, 0, 1)
    elif defect == "signed_zero":
        edit(output, 2, 0x8000, bits=True)
        # Decode does not claim arithmetic reconstruction; its finite output may
        # pass. Promotion must still bind its actual source and preserve -0.
    capture.capture_wk(state.root, state.record, state.bound.local_slots, layer=0,
        name="wk_decode", value=output, inputs=(raw.wk_bits_local, raw.wk_scale_local), platform="cpu") if defect in ("wk_link", "signed_zero") else None
    if defect in ("nonfinite", "checkpoint"):
        with pytest.raises(ValueError, match="originals preserved"):
            capture.capture_wk(state.root, state.record, state.bound.local_slots, layer=0,
                name="wk_decode", value=output, inputs=(raw.wk_bits_local, raw.wk_scale_local), platform="cpu")
        name, role = "wk_decode", "result" if defect == "nonfinite" else "input0"
    else:
        if defect == "wk_link":
            edit(output, 3, 1)
        with pytest.raises(ValueError, match="originals preserved"):
            capture.capture_wk(state.root, state.record, state.bound.local_slots, layer=0,
                name="wk_promote", value=state.promoted[0].wk_weight, inputs=(output,), platform="cpu")
        name, role = "wk_promote", "input0"
    report = state.record["materializer_originals"][f"layer0/{name}"]
    assert not report["valid"] and report["errors"]
    with np.load(state.root / report["original"]["path"], allow_pickle=False) as saved:
        actual = getattr(raw, "wk_bits_local") if defect == "checkpoint" else output
        assert saved[f"slot9_{role}"].tobytes() == actual.addressable_shards[0].data.tobytes()


@pytest.mark.parametrize("defect", ["output_nonfinite", "input_nonfinite", "checkpoint", "wk_link"])
def test_exact_decode_bounded_refused_original_and_complete_receipt(state, defect):
    wk_calls(state)
    if defect == "output_nonfinite":
        changed = state.decoded[2].qkv_a_scale
        edit(changed, 10, 0x7FC01234, bits=True)
    elif defect == "input_nonfinite":
        changed = state.raw[2].q_a_scale_local
        edit(changed, 10, 0x7FC04321, bits=True)
    elif defect == "checkpoint":
        changed = state.raw[2].q_a_bits_local
        edit(changed, 10, 1)
    else:
        changed = state.decoded[2].wk_weight_bf16
        edit(changed, 10, 0x8000, bits=True)
    with pytest.raises(ValueError, match="bounded completed originals preserved"):
        exact(state)
    report = capture.load_exact_receipt(state.root, state.record, "exact_decode")
    assert not report["valid"] and len(report["leaves"]) == 20 and len(report["source_leaves"]) == 36
    refusal = report["refused_original"]
    with np.load(state.root / refusal["original"]["path"], allow_pickle=False) as saved:
        assert saved["sample0"].tobytes() == changed.addressable_shards[0].data.tobytes()
        assert len(saved.files) == 1
    assert refusal["original"]["bytes"] < capture.REFUSAL_ORIGINAL_LIMIT


@pytest.mark.parametrize("field", ["qkv_a_bits", "qkv_a_scale", "wq_b_weight_aliases/3", "wk_weight", "head_weight_local", "input_replacement"])
def test_promotion_relations_and_original_input_chain(state, field):
    wk_calls(state)
    exact(state)
    changed = state.decoded[0].wq_b_weight_local if field == "input_replacement" else capture._get(state.promoted[0], field)
    edit(changed, 0, 2)
    with pytest.raises(ValueError, match="bounded completed originals preserved"):
        exact(state, "exact_promote")
    report = capture.load_exact_receipt(state.root, state.record, "exact_promote")
    assert not report["valid"] and len(report["leaves"]) == 32
    suffix = "exact_decode_original_bytes" if field == "input_replacement" else "promotion_bytes"
    assert any(error.endswith(suffix) for error in report["errors"])


@pytest.mark.parametrize("defect", ["signed_zero", "nan_payload", "uint8"])
def test_conflicting_replica_pair_bytes_are_saved_without_full_tree(tmp_path, defect):
    slots = {12: 9, 14: 13}
    raw, decoded, promoted = tree(capture.RAW, slots), tree(capture.DECODED, slots), tree(capture.PROMOTED, slots)
    state = NS(root=tmp_path, bound=NS(raw_exact=raw, local_slots=slots), record=record(raw, slots),
               raw=raw, decoded=decoded, promoted=promoted)
    wk_calls(state)
    if defect == "uint8":
        changed = decoded[0].qkv_a_bits
        edit(changed, 0, 1, owner=1)
    else:
        changed = decoded[0].qkv_a_scale
        edit(changed, 0, 0x7FC01234 if defect == "nan_payload" else 0, owner=0, bits=True)
        edit(changed, 0, 0x7FC04321 if defect == "nan_payload" else 0x80000000, owner=1, bits=True)
    with pytest.raises(ValueError, match="originals preserved"):
        exact(state)
    report = capture.load_exact_receipt(tmp_path, state.record, "exact_decode")
    refused = report["refused_original"]
    assert len(refused["samples"]) == 2
    with np.load(tmp_path / refused["original"]["path"], allow_pickle=False) as saved:
        for i in range(2):
            assert saved[f"sample{i}"].tobytes() == changed.addressable_shards[i].data.tobytes()
    assert refused["original"]["raw_array_bytes"] <= capture.MAX_REFUSAL_RAW_BYTES
    assert not (tmp_path / "materializers/exact_decode.npz").exists()


@pytest.mark.parametrize("defect", ["overwrite", "symlink", "unexpected", "wk_budget", "total_budget", "json_budget", "refusal_budget"])
def test_exclusive_original_paths_and_budget_guards(state, defect, monkeypatch):
    arrays = {"sample0": np.zeros(8, np.float32)}
    if defect == "overwrite":
        first = capture._npz(state.root, "refused_exact", arrays)
        before = (state.root / first["path"]).read_bytes()
        with pytest.raises(FileExistsError):
            capture._npz(state.root, "refused_exact", arrays)
        assert (state.root / first["path"]).read_bytes() == before
        return
    if defect == "symlink":
        (state.root / "materializers").symlink_to(state.root, target_is_directory=True)
    elif defect == "unexpected":
        directory = state.root / "materializers"
        directory.mkdir()
        (directory / "unknown").touch()
    elif defect == "wk_budget":
        monkeypatch.setattr(capture, "WK_ORIGINALS_LIMIT", 1)
    elif defect == "total_budget":
        monkeypatch.setattr(capture, "TOTAL_LIMIT", 1)
    elif defect == "json_budget":
        monkeypatch.setattr(capture, "EXACT_RECEIPT_LIMIT", 1)
    else:
        monkeypatch.setattr(capture, "REFUSAL_ORIGINAL_LIMIT", 1)
    with pytest.raises(ValueError):
        if defect == "json_budget":
            capture._json(state.root, "exact_decode", {"key": "value"})
        else:
            capture._npz(state.root, "layer0/wk_decode" if defect == "wk_budget" else "refused_exact", arrays)


def test_writer_keeps_partial_original_bounded_without_pending_copy(tmp_path):
    path = tmp_path / "partial"
    with path.open("xb") as stream:
        writer = capture._CappedWriter(stream, 3)
        writer.write(b"abc")
        with pytest.raises(ValueError, match="reserved bytes"):
            writer.write(b"d")
    assert path.read_bytes() == b"abc"


@pytest.mark.parametrize("defect", ["hash", "size", "path", "identity", "valid", "symlink"])
def test_independent_exact_receipt_resolution_refuses_tampering(state, defect):
    report = capture._base("exact_decode", state.bound.local_slots, state.record)
    report["valid"] = True
    ref = capture._json(state.root, "exact_decode", report)
    state.record["materializer_originals"] = {"exact_decode": dict(report, original=ref)}
    path = state.root / ref["path"]
    if defect == "hash":
        ref["sha256"] = "0" * 64
    elif defect == "size":
        ref["bytes"] += 1
    elif defect == "path":
        ref["path"] = "../exact_decode.json"
    elif defect == "identity":
        report["name"] = "exact_promote"
        path.write_text(json.dumps(report))
        ref.update(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())
    elif defect == "valid":
        state.record["materializer_originals"]["exact_decode"]["valid"] = False
    else:
        path.unlink()
        path.symlink_to(state.root / "missing")
    with pytest.raises(ValueError):
        capture.load_exact_receipt(state.root, state.record, "exact_decode")


def test_default_requires_actual_four_tpu_owners(state):
    with pytest.raises(ValueError, match="owner map"):
        capture.capture_wk(state.root, state.record, state.bound.local_slots, layer=0,
            name="wk_decode", value=state.decoded[0].wk_weight_bf16,
            inputs=(state.raw[0].wk_bits_local, state.raw[0].wk_scale_local))


def test_source_names_match_frozen_selector():
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DecoderConfig, ws32_decoder_weight_names, select_ws32_exact_dsa_raw_weights,
    )
    from glm_tpu.greenfield.types import ModelGeometry
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    geometry = ModelGeometry.from_hf_config(json.loads((repo / "configs/glm-5.2-fp8-config.json").read_text()))
    config = Ws32DecoderConfig(geometry, 8192, host_main_rope_table=True, exact_dsa=True)
    names = ws32_decoder_weight_names(config)
    selected = select_ws32_exact_dsa_raw_weights(names, config)
    for layer in protocol.PRODUCERS:
        index = config.full_index_slots.index(layer)
        assert capture.source_names(layer) == selected[index]._asdict()


def test_declared_contract_bytes_match_admitted_four_producer_interfaces():
    from math import prod
    for specs, expected in ((capture.RAW, 21_156_992), (capture.DECODED, 106_741_760),
                            (capture.PROMOTED, 213_696_512)):
        per_layer = 0
        for _, shape, dtype, axes in specs:
            index = capture._index(shape, axes, 9)
            per_layer += prod(len(range(*part.indices(size))) for part, size in zip(index, shape, strict=True)) * np.dtype(dtype).itemsize
        assert per_layer * 4 == expected


def test_real_cpu_completed_wk_boundaries_are_preserved(tmp_path):
    import jax
    import jax.numpy as jnp
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16, promote_stage_local_prefill_index_wk,
    )
    assert jax.default_backend() == "cpu"
    device = jax.local_devices()[0]
    slots = {device.id: 9}
    raw = tree(capture.RAW, slots)
    bits = jnp.full((128, 6144), 0x30, jnp.uint8)
    scales = jnp.full((1, 48), 0.75, jnp.float32)
    for layer in raw:
        for field in layer._fields:
            getattr(layer, field).addressable_shards[0].device = device
        layer.wk_bits_local.addressable_shards[0].data = bits[:, 1536:3072]
        layer.wk_scale_local.addressable_shards[0].data = scales[:, 12:24]
    runtime_record = record(raw, slots)
    runtime_record["jax_process_index"] = device.process_index
    decoded = jax.jit(decode_stage_local_prefill_index_wk_bf16)(bits, scales).block_until_ready()
    promoted = jax.jit(promote_stage_local_prefill_index_wk)(decoded).block_until_ready()
    capture.capture_wk(tmp_path, runtime_record, slots, layer=0, name="wk_decode",
        value=decoded, inputs=(raw[0].wk_bits_local, raw[0].wk_scale_local), platform="cpu")
    capture.capture_wk(tmp_path, runtime_record, slots, layer=0, name="wk_promote",
        value=promoted, inputs=(decoded,), platform="cpu")
    reference = runtime_record["materializer_originals"]["layer0/wk_promote"]["original"]
    with np.load(tmp_path / reference["path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["slot9_result"], np.asarray(decoded).astype(np.float32))
        assert saved["slot9_input0"].tobytes() == np.asarray(decoded).tobytes()
    assert not list(tmp_path.glob("**/*.pending"))
