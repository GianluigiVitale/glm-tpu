"""Addressable-only capture on fake multi-host arrays, without JAX compilation."""

import json
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_history_capture as capture
from scripts.greenfield.ws32_history_frontier import LayerBoundary


SLOTS = {29: 31, 9: 0, 25: 8, 13: 19}
PRODUCTION_SLOTS = {12: 9, 14: 13, 13: 25, 15: 29}
COMMON = dict(local_slots=SLOTS, process_index=3, platform="tpu")


class GlobalArray:
    is_fully_addressable = False

    def __init__(self, shape, dtype, shards):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.addressable_shards = shards

    def __array__(self, *args, **kwargs):
        raise AssertionError("attempted a global array read")


def array(shape, dtype, slots=SLOTS, *, kind="replica", platform="tpu", value=0):
    shards = []
    for device, slot in slots.items():
        index = [slice(None)] * len(shape)
        if kind == "feature":
            width = shape[1] // 4
            index[1] = slice(slot % 4 * width, (slot % 4 + 1) * width)
        elif kind == "health":
            index[:2] = [slice(slot // 4, slot // 4 + 1), slice(slot % 4, slot % 4 + 1)]
        local_shape = tuple(len(range(*s.indices(size))) for s, size in zip(index, shape))
        data = np.full(local_shape, value, dtype)
        if kind == "feature":
            data += np.arange(index[1].start, index[1].stop, dtype=np.float32).astype(dtype)[None]
        shards.append(NS(device=NS(id=device, process_index=3, platform=platform),
                         index=tuple(index), data=data))
    return GlobalArray(shape, dtype, shards)


def result(*, physical_rows=128, hidden=16, slots=SLOTS, platform="tpu"):
    boundaries = []
    for layer in range(7):
        kwargs = dict(slots=slots, platform=platform)
        boundaries.append(LayerBoundary(
            *(array((physical_rows, hidden), ml_dtypes.bfloat16,
                    kind="feature", value=layer, **kwargs) for _ in range(3)),
            array((physical_rows, 8), np.int32, value=layer, **kwargs),
            array((physical_rows, 8), np.float32, value=layer / 8, **kwargs),
            array((8, 4, physical_rows), np.bool_, kind="health", value=True, **kwargs),
        ))
    return NS(boundaries=tuple(boundaries))


def test_partial_feature_slices_are_sorted_deduplicated_and_explicit():
    rows, layout, errors = capture.capture_boundaries(result(), 3, **COMMON)
    assert not errors
    assert layout["feature_columns"] == [0, 1, 2, 3, 12, 13, 14, 15]
    assert layout["health_slots"] == [0, 8, 19, 31]
    assert layout["local_slots"] == {str(d): s for d, s in SLOTS.items()}
    assert layout["owners"]["19"] == dict(device_id=13, slot=19,
        feature_columns=[12, 13, 14, 15], capture_columns=[4, 5, 6, 7])
    assert json.loads(json.dumps(layout)) == layout
    for layer in range(7):
        for field in capture.FEATURE_FIELDS:
            expected = np.tile(np.asarray(layout["feature_columns"]) + layer, (3, 1))
            np.testing.assert_array_equal(rows[layer][field], expected)
            assert rows[layer][field].dtype == ml_dtypes.bfloat16
        assert rows[layer]["route_ids"].shape == (3, 8)
        assert rows[layer]["route_weights"].dtype == np.float32
        assert rows[layer]["health"].shape == (3, 4)
        assert rows[layer]["health"].all()


def test_actual_rank_zero_mapping_has_only_feature_one_with_four_replicas():
    observed = result(physical_rows=1, hidden=6144, slots=PRODUCTION_SLOTS)
    rows, layout, errors = capture.capture_boundaries(observed, 1,
        local_slots=PRODUCTION_SLOTS, process_index=3, platform="tpu")
    assert not errors
    assert rows[0]["update"].shape == (1, 1536)
    assert layout["feature_columns"] == list(range(1536, 3072))
    assert layout["health_slots"] == [9, 13, 25, 29]
    for slot in PRODUCTION_SLOTS.values():
        assert layout["owners"][str(slot)]["feature_columns"] == list(range(1536, 3072))
        assert layout["owners"][str(slot)]["capture_columns"] == list(range(1536))


def test_cpu_full_mesh_and_layout_independent_of_rows():
    slots = {100 + slot: 31 - slot for slot in range(32)}
    common = dict(local_slots=slots, process_index=3, platform="cpu")
    layouts = []
    for physical_rows, count in ((128, 32), (114, 27), (1, 1)):
        rows, layout, errors = capture.capture_boundaries(
            result(physical_rows=physical_rows, slots=slots, platform="cpu"), count, **common)
        assert not errors
        assert rows[0]["update"].shape == (count, 16)
        assert rows[0]["health"].shape == (count, 32)
        layouts.append(layout)
    assert layouts[0] == layouts[1] == layouts[2]
    np.testing.assert_array_equal(rows[0]["update"][0], np.arange(16))


@pytest.mark.parametrize("shape,dtype,value", [
    ((), np.bool_, False), ((), np.int32, 4), ((4,), np.int32, 2),
    ((4, 1, 2048), np.float32, -np.inf), ((4, 1), np.int32, 2048),
])
def test_replicated_scalar_and_dsa_read_only_local_shards(shape, dtype, value):
    distributed = array(shape, dtype, value=value)
    host = capture.replicated_host(distributed, **COMMON)
    assert host.shape == shape and host.dtype == dtype
    assert (host == value).all()
    host[...] = 0
    assert (distributed.addressable_shards[0].data == value).all()


@pytest.mark.parametrize("field", [*capture.FEATURE_FIELDS, "route_weights"])
def test_nonfinite_live_stream_bytes_are_returned_before_refusal(field):
    observed = result()
    # All replicas of this feature must retain the same original NaN bits.
    for shard in getattr(observed.boundaries[2], field).addressable_shards:
        shard.data[1, 0] = np.nan
    rows, _, errors = capture.capture_boundaries(observed, 3, **COMMON)
    assert errors == [f"layer2/{field}/nonfinite"]
    assert np.isnan(rows[2][field][1, 0])


def test_every_health_owner_is_checked_and_preserved_in_slot_order():
    observed = result()
    observed.boundaries[6].health.addressable_shards[0].data[0, 0, 2] = False
    rows, layout, errors = capture.capture_boundaries(observed, 3, **COMMON)
    assert errors == ["layer6/slot31/health"]
    assert not rows[6]["health"][2, layout["health_slots"].index(31)]
    assert rows[6]["health"].sum() == 11


def test_dead_padding_does_not_raise_live_health_or_numerical_error():
    observed = result()
    for shard in observed.boundaries[0].update.addressable_shards:
        shard.data[100, 0] = np.nan
    for shard in observed.boundaries[0].health.addressable_shards:
        shard.data[0, 0, 100] = False
    _, _, errors = capture.capture_boundaries(observed, 32, **COMMON)
    assert not errors


@pytest.mark.parametrize("field", ["update", "route_ids", "route_weights"])
def test_inconsistent_replicas_are_never_silently_deduplicated(field):
    observed = result()
    getattr(observed.boundaries[0], field).addressable_shards[0].data[0, 0] = 100
    with pytest.raises(capture.ReplicaMismatch, match="replica bytes") as raised:
        capture.capture_boundaries(observed, 1, **COMMON)
    error = raised.value
    assert set(error.originals) == {"replica0", "replica1"}
    assert error.metadata["context"] == f"layer0/{field}"
    assert error.metadata["replicas"]["replica1"]["device_id"] == 29
    assert error.metadata["replicas"]["replica1"]["slot"] == 31
    assert error.originals["replica1"][0, 0] == 100
    assert error.originals["replica0"][0, 0] != 100
    assert error.originals["replica0"].shape[0] == 128  # Exact whole replica, including padding.
    assert json.loads(json.dumps(error.metadata)) == error.metadata
    originals = {shard.device.id: shard.data for shard in getattr(observed.boundaries[0], field).addressable_shards}
    for name, value in error.originals.items():
        assert value.tobytes() == originals[error.metadata["replicas"][name]["device_id"]].tobytes()
    # The refusal payload owns its copies even if the source is later reused.
    originals[29][0, 0] = 101
    assert error.originals["replica1"][0, 0] == 100


def test_replicated_signed_zero_difference_refuses_but_identical_nan_bits_copy():
    distributed = array((1,), np.float32)
    distributed.addressable_shards[-1].data[0] = -0.0
    with pytest.raises(capture.ReplicaMismatch, match="replica bytes") as raised:
        capture.replicated_host(distributed, context="observer/scores", **COMMON)
    error = raised.value
    assert error.originals["replica0"].view(np.uint32).tolist() == [0]
    assert error.originals["replica1"].view(np.uint32).tolist() == [0x80000000]
    assert error.metadata == dict(context="observer/scores", global_shape=[1], shape=[1], dtype="float32",
        replicas={"replica0": dict(device_id=9, slot=0, index=[[0, 1, 1]]),
                  "replica1": dict(device_id=13, slot=19, index=[[0, 1, 1]])})
    for shard in distributed.addressable_shards:
        shard.data[0] = np.nan
    assert np.isnan(capture.replicated_host(distributed, **COMMON)[0])


def test_distinct_nan_payloads_are_preserved_exactly_in_refusal():
    distributed = array((1,), np.float32)
    for shard in distributed.addressable_shards:
        shard.data.view(np.uint32)[0] = 0x7FC00001
    distributed.addressable_shards[-1].data.view(np.uint32)[0] = 0x7FC00002
    with pytest.raises(capture.ReplicaMismatch) as raised:
        capture.replicated_host(distributed, **COMMON)
    assert raised.value.originals["replica0"].view(np.uint32).tolist() == [0x7FC00001]
    assert raised.value.originals["replica1"].view(np.uint32).tolist() == [0x7FC00002]
    assert all(np.isnan(value).all() for value in raised.value.originals.values())


def test_replica_refusal_keeps_only_first_conflicting_pair_and_stops_reading():
    class Unreadable:
        shape, dtype = (1,), np.dtype(np.float32)

        def __array__(self, *args, **kwargs):
            raise AssertionError("downloaded another replica after the first conflict")

    distributed = array((1,), np.float32)
    by_device = {shard.device.id: shard for shard in distributed.addressable_shards}
    by_device[13].data[0] = 1
    by_device[25].data = Unreadable()
    with pytest.raises(capture.ReplicaMismatch) as raised:
        capture.replicated_host(distributed, **COMMON)
    assert len(raised.value.originals) == 2
    assert sum(value.nbytes for value in raised.value.originals.values()) == 8


def test_production_feature_replica_refusal_is_bounded_and_has_actual_indices():
    observed = result(hidden=6144, slots=PRODUCTION_SLOTS)
    observed.boundaries[0].update.addressable_shards[-1].data[100, 0] = -1
    with pytest.raises(capture.ReplicaMismatch) as raised:
        capture.capture_boundaries(observed, 32,
            local_slots=PRODUCTION_SLOTS, process_index=3, platform="tpu")
    error = raised.value
    assert error.metadata["shape"] == [128, 1536]
    assert error.metadata["global_shape"] == [128, 6144]
    assert error.metadata["replicas"] == {
        "replica0": dict(device_id=12, slot=9, index=[[0, 128, 1], [1536, 3072, 1]]),
        "replica1": dict(device_id=15, slot=29, index=[[0, 128, 1], [1536, 3072, 1]]),
    }
    assert len(error.originals) == 2
    assert sum(value.nbytes for value in error.originals.values()) == 786432
    assert error.originals["replica1"][100, 0] == -1


@pytest.mark.parametrize("mutation", [
    "device", "device_type", "platform", "process", "missing", "duplicate",
    "feature_index", "row_index", "rank_dropping_index", "health_expert", "health_feature",
    "global_shape", "global_dtype", "local_shape", "local_dtype",
])
def test_actual_owner_and_geometry_mutations_refuse(mutation):
    observed = result()
    leaf = observed.boundaries[0].health if mutation.startswith("health") else observed.boundaries[0].update
    shard = leaf.addressable_shards[0]
    if mutation == "device":
        shard.device.id = 987
    elif mutation == "device_type":
        shard.device.id = str(shard.device.id)
    elif mutation == "platform":
        shard.device.platform = "cpu"
    elif mutation == "process":
        shard.device.process_index = 4
    elif mutation == "missing":
        leaf.addressable_shards.pop()
    elif mutation == "duplicate":
        leaf.addressable_shards.append(shard)
    elif mutation == "feature_index":
        shard.index = (slice(None), slice(0, 4))
    elif mutation == "row_index":
        shard.index = (slice(1, 128), *shard.index[1:])
    elif mutation == "rank_dropping_index":
        shard.index = (0, *shard.index[1:])
    elif mutation == "health_expert":
        shard.index = (slice(0, 1), *shard.index[1:])
    elif mutation == "health_feature":
        shard.index = (shard.index[0], slice(0, 1), shard.index[2])
    elif mutation == "global_shape":
        leaf.shape = (127, 16)
    elif mutation == "global_dtype":
        leaf.dtype = np.dtype(np.float32)
    elif mutation == "local_shape":
        shard.data = shard.data[:127]
    elif mutation == "local_dtype":
        shard.data = shard.data.astype(np.float32)
    with pytest.raises(ValueError):
        capture.capture_boundaries(observed, 1, **COMMON)


def test_all_leaf_metadata_checked_before_any_read():
    class Unreadable:
        def __init__(self, value):
            self.shape, self.dtype = value.shape, value.dtype

        def __array__(self, *args, **kwargs):
            raise AssertionError("read a payload before validating all boundary leaves")

    observed = result()
    shard = observed.boundaries[0].update.addressable_shards[0]
    shard.data = Unreadable(shard.data)
    observed.boundaries[-1].health.addressable_shards.pop()
    with pytest.raises(ValueError, match="missing addressable"):
        capture.capture_boundaries(observed, 1, **COMMON)


def test_replicated_reader_rejects_sharded_array_and_bad_replica_geometry():
    with pytest.raises(ValueError, match="shard index"):
        capture.replicated_host(array((128, 16), ml_dtypes.bfloat16, kind="feature"), **COMMON)
    distributed = array((4, 1), np.int32)
    distributed.addressable_shards[-1].data = np.zeros((4, 1), np.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        capture.replicated_host(distributed, **COMMON)


@pytest.mark.parametrize("overrides", [
    {"local_slots": {}}, {"local_slots": {9: 0}}, {"local_slots": {9: 0, 13: 0, 25: 8, 29: 31}},
    {"local_slots": {9: 0, 13: 1, 25: 8, 29: 32}}, {"process_index": True}, {"platform": "gpu"},
])
def test_bad_authenticated_mapping_or_runtime_identity_refuses(overrides):
    with pytest.raises(ValueError, match="owner map or runtime identity"):
        capture.capture_boundaries(result(), 1, **{**COMMON, **overrides})


@pytest.mark.parametrize("count", [0, -1, True, 129])
def test_bad_live_count_refuses(count):
    with pytest.raises(ValueError):
        capture.capture_boundaries(result(), count, **COMMON)
