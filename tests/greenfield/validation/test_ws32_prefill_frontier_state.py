"""Host ownership/metadata/failure originals; no hardware admission claim."""

from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield.ws32_prefill_frontier import capture_cache_evidence
from scripts.greenfield.ws32_prefill_frontier_state import (
    INDEX_LAYERS, METADATA, capture_state, metadata_errors, owner_shards,
)


def metadata(frontier):
    values = {name: np.zeros(shape, dtype) for name, (shape, dtype) in METADATA.items()}
    values["position"][:] = frontier
    values["context_lengths"][:] = frontier+1
    values["prompt_length"][()] = 8155
    values["contract_valid"][:] = True
    values["next_token"][:] = -1
    values["selected_valid_counts"][:] = frontier
    values["block_tables"][:] = np.arange(16)
    values["selected_positions"][:] = -1
    values["selected_positions"][0, :frontier] = np.arange(frontier)
    values["selected_scores"][:] = -np.inf
    values["selected_scores"][0, :frontier] = np.arange(frontier)
    return values


@pytest.mark.parametrize("frontier", [0, 32, 64, 96, 128])
def test_exact_frontiers(frontier):
    assert not metadata_errors(metadata(frontier), frontier=frontier)


@pytest.mark.parametrize("field,value", [
    ("position", 127), ("context_lengths", 128), ("prompt_length", 128),
    ("contract_valid", False), ("finished", True), ("next_token", 3),
    ("selected_valid_counts", 32), ("block_tables", 0),
    ("selected_positions", 0), ("selected_scores", np.nan),
])
def test_bad_metadata_remains_explicit(field, value):
    values = metadata(128)
    values[field][...] = value
    assert metadata_errors(values, frontier=128)


def test_raw_failure_bits_are_retained_with_bounded_samples():
    value = np.zeros((2, 16, 64, 128), ml_dtypes.bfloat16)
    value[0, 0, 0, 0] = np.inf
    value[1, 9, :, 1] = -0.0
    rows, record = capture_cache_evidence(
        value, slot=0, block_table=np.arange(16, dtype=np.int32)[None], layer_ids=(0, 6))
    assert not record["valid"] and record["samples_truncated"]
    assert record["violations"] == dict(nonfinite=1, outside_nonzero=64)
    assert len(record["samples"]) == 32
    assert record["samples"][0]["bits"] == 0x7F80
    assert record["samples"][1]["bits"] == 0x8000
    assert rows[0, 0, 0] == 0x7F80
    assert not record["numerical_promotion"]


def test_initial_rejects_written_rows_even_inside_first128():
    value = np.zeros((1, 1, 64, 128), ml_dtypes.bfloat16)
    value[0, 0, 2, 3] = -0.0
    _, record = capture_cache_evidence(
        value, slot=0, block_table=np.asarray([[0]], np.int32), layer_ids=(0,), initial=True)
    assert not record["valid"] and record["violations"]["outside_nonzero"] == 1
    assert record["samples"][0]["local_row"] == 2


def owner_array():
    slots = {91: 4, 17: 1, 32: 0, 4: 7}  # Deliberately neither rank nor iterator order.
    shards = [NS(device=NS(id=d, process_index=3, platform="tpu"),
                 index=(slice(None), slice(None), slice((s//4)*64, (s//4+1)*64), slice(None)),
                 data=np.zeros((1, 1, 64, 128), ml_dtypes.bfloat16))
              for d, s in slots.items()]
    return NS(shape=(1, 1, 512, 128), dtype=np.dtype(ml_dtypes.bfloat16),
              addressable_shards=shards), slots


@pytest.mark.parametrize("mutation", [None, "stripe", "process", "missing", "duplicate", "platform"])
def test_actual_owner_identity_and_stripe(mutation):
    array, slots = owner_array()
    shard = array.addressable_shards[0]
    if mutation == "stripe":
        shard.index = (slice(None), slice(None), slice(0, 64), slice(None))
    if mutation == "process":
        shard.device.process_index = 0
    if mutation == "missing":
        array.addressable_shards.pop()
    if mutation == "duplicate":
        array.addressable_shards.append(shard)
    if mutation == "platform":
        shard.device.platform = "cpu"
    call = lambda: owner_shards(array, shape=array.shape, dtype="bfloat16", cache=True,
                               local_slots=slots, process_index=3)
    if mutation:
        with pytest.raises(ValueError):
            call()
    else:
        assert set(call()) == set(slots)


def test_production_state_capture_preserves_failed_owner_originals():
    slots = {91: 0, 17: 1, 32: 2, 4: 3}
    config = NS(context_capacity=8192, logical_page_size=512, local_rows_per_page=64,
                kv_cache_shape=(78, 16, 512, 640), index_cache_shape=(21, 16, 512, 128),
                full_index_slots=INDEX_LAYERS, exact_dsa=False, strategy_nd_dense=False,
                host_main_rope_table=True, geometry=NS(hidden_size=6144, dsa_top_k=2048))
    def distributed(values, shape, cache=False):
        assert set(values) == set(slots)
        index = [slice(None)] * len(shape)
        if cache:
            index[2] = slice(0, 64)
        return NS(shape=shape, dtype=next(iter(values.values())).dtype,
                  addressable_shards=[NS(device=NS(id=d, process_index=3, platform="tpu"),
                                         index=tuple(index), data=v) for d, v in values.items()])
    host = metadata(128)
    fields = {name: distributed({d: value.copy() for d in slots}, value.shape)
              for name, value in host.items()}
    def cache(shape):
        local_shape = (*shape[:2], 64, shape[3])
        # Real production host shape with compact fixture allocation; capture
        # still reads/hashes/reconstructs every observed cache byte.
        zero = np.broadcast_to(np.asarray(0, ml_dtypes.bfloat16), local_shape)
        return distributed({d: zero for d in slots}, shape, True)
    kv = cache(config.kv_cache_shape)
    damaged = kv.addressable_shards[0].data.copy()
    damaged[77, 9, 4, 2] = -0.0  # Hidden future write on just one feature replica.
    kv.addressable_shards[0].data = damaged
    decoder = NS(**{name: value for name, value in fields.items()
                    if name not in ("prompt_length", "finished", "next_token")},
                 kv_cache_local=kv, index_cache_local=cache(config.index_cache_shape))
    state = NS(decoder=decoder, repaired_index_local=cache(config.index_cache_shape),
               prompt_length=fields["prompt_length"], finished=fields["finished"])
    arrays, report = capture_state(state, next_token=fields["next_token"], config=config,
                                   local_slots=slots, process_index=3, frontier=128)
    assert not report["valid"]
    assert "slot0:kv:cache_bytes" in report["errors"]
    evidence = report["owners"]["0"]["caches"]["kv"]
    assert evidence["samples"][0] == dict(layer_id=77, physical_page=9, local_row=4,
        component=2, bits=32768, nonfinite=False, outside_nonzero=True)
    assert arrays["slot0_kv_rows"].shape == (78, 64, 640)
    assert sum(a.nbytes for a in arrays.values()) < 32*1024**2
    assert not report["numerical_promotion"]
