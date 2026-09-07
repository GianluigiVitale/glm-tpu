"""CPU-only original-evidence/worker plumbing, not TPU admission."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
from scripts.greenfield.prefill_layer_evidence import (
    BF16,
    FIELDS,
    INPUT_FIELDS,
    METAMORPHIC,
    check_intervention,
    decode_arrays,
    encode_arrays,
    equal_bytes,
    host_case,
    local_observations,
    mutate_case,
    owner_inputs,
    replay_case,
    stack_reference,
)
from scripts.greenfield.prefill_layer_numerical import CASES, ROWS, written_addresses
from scripts.greenfield.probe_ws32_prefill_layer import layer_from_tag, scalar_inputs


def host(case):
    return host_case(case, build_rotary_table_host(1024, rotary_dim=64, theta=8e6))


def observations(inputs, *, slot, layer):
    own = owner_inputs(inputs, slot)
    count = int(inputs["count"])
    values = {
        n: own[n].copy()
        for n in ("kv", "index", "repair", "positions", "counts", "scores", "health")
    }
    for name in ("output", "residual", "normalized"):
        values[name] = np.zeros((ROWS, 1536), BF16)
        values[name][:count] = 0.01
    values["routes"] = np.full((ROWS, 8), -1, np.int32)
    values["route_weights"] = np.zeros((ROWS, 8), np.float32)
    if layer == 3:
        values["routes"][:count] = np.arange(8, dtype=np.int32)
        values["route_weights"][:count] = 0.125
    keys = np.full((ROWS, 128), 0.03, BF16)
    for _, page, row in written_addresses(
        slot=slot, offset=int(inputs["offset"]), count=count
    ):
        values["kv"][page, row, :576] = 0.02
        if layer == 0:
            values["index"][page, row] = 0.04
            values["repair"][page, row] = 0.03
    return values, keys


def evidence(case, layer, slot=0):
    inputs = host(case)
    actual, keys = observations(inputs, slot=slot, layer=layer)
    arrays = encode_arrays("input", inputs)
    for kind in ("actual", "reference"):
        arrays.update(encode_arrays(kind + "_9", actual))
        if layer == 0:
            arrays.update(encode_arrays(kind + "_9", {"m64": keys}))
    for kind in METAMORPHIC[case]:
        changed = deepcopy(actual)
        modified = mutate_case(inputs, kind)
        if kind in ("bad_offset", "bad_page"):
            changed["health"][:] = False
            for name in ("kv", "index", "repair"):
                changed[name] = owner_inputs(inputs, slot)[name].copy()
        elif kind == "incoming_health":
            changed["health"][0] = False
        elif kind == "repair_history":
            changed["repair"] = owner_inputs(modified, slot)["repair"].copy()
            for _, p, r in written_addresses(
                slot=slot, offset=int(inputs["offset"]), count=int(inputs["count"])
            ):
                changed["repair"][p, r] = actual["repair"][p, r]
        arrays.update(encode_arrays(kind + "_9", changed))
    return arrays


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("layer", [0, 3])
def test_original_array_roundtrip_and_replay(tmp_path, case, layer):
    path = tmp_path / "original.npz"
    arrays = evidence(case, layer)
    np.savez_compressed(path, **arrays)
    result = replay_case(path, layer=layer, case=case, slots_by_device={9: 0})
    assert result["passed"]
    assert set(result["owners"]["9"]["interventions"]) == set(METAMORPHIC[case])


@pytest.mark.parametrize("field", INPUT_FIELDS)
def test_fixed_inputs_cannot_be_replaced_by_claimed_hash(tmp_path, field):
    arrays = evidence("tail", 0)
    # Raw bytes, including padded NaNs, are fixed by the protocol, not the worker.
    arrays["input__" + field] = arrays["input__" + field].copy()
    value = arrays["input__" + field]
    value.reshape(-1)[0] = 0 if value.reshape(-1)[0] != 0 else 1
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="original synthetic inputs"):
        replay_case(path, layer=0, case="tail", slots_by_device={9: 0})


@pytest.mark.parametrize("kind", METAMORPHIC["boundary"] + METAMORPHIC["tail"])
def test_metamorphic_claims_need_actual_bytes(tmp_path, kind):
    case = "tail" if kind == "clean_padding" else "boundary"
    arrays = evidence(case, 0)
    field = (
        "health" if kind in ("bad_offset", "bad_page", "incoming_health") else "output"
    )
    target = f"{kind}_9__{field}"
    arrays[target] = arrays[target].copy()
    arrays[target].reshape(-1)[0] = True if field == "health" else 0
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    assert not replay_case(path, layer=0, case=case, slots_by_device={9: 0})["passed"]


def test_host_fixture_reordered_pages_and_padding():
    inputs = host("tail")
    assert np.isnan(inputs["update"][11:].astype(np.float32)).all()
    assert not np.any(inputs["kv"][:, 0])  # Reordered page0 is the unwritten future.
    assert np.any(inputs["kv"][7, 1, 56]) and not np.any(inputs["kv"][7, 1, 57:])
    assert np.all(inputs["kv"][:, :, :, 576:] == 0)
    assert all(equal_bytes(inputs[n], host("tail")[n]) for n in INPUT_FIELDS)
    own = owner_inputs(inputs, 29)
    assert equal_bytes(own["update"], inputs["update"][:, 1536:3072])
    assert equal_bytes(own["kv"], inputs["kv"][7])


def test_scalar_input_carry_is_reference_only():
    import jax.numpy as jnp

    host_values = host("boundary")
    values = (
        tuple(jnp.asarray(host_values[n]) for n in INPUT_FIELDS[:11])
        + (None,) * 7
        + (jnp.asarray(host_values["health"]), jnp.asarray(host_values["rope"]))
    )
    previous = [None] * 12
    previous[2] = jnp.ones_like(values[2])
    previous[3] = jnp.ones_like(values[3])
    one = scalar_inputs(values, 8, tuple(previous))
    assert int(one[8]) == 513 and int(one[9]) == 1
    assert one[0].shape == (1, 6144) and one[18].shape == (8, 4, 1)
    assert one[2] is previous[2] and one[3] is previous[3] and one[4] is values[4]
    np.testing.assert_array_equal(one[5], values[5][8:9])
    assert not np.all(np.asarray(values[2]) == 1)


def test_stack_reference_keeps_final_cache_and_pads_rows():
    inputs = host("tail")
    actual, _ = observations(inputs, slot=0, layer=0)
    rows = []
    for i in range(11):
        rows.append(
            {
                n: (
                    actual[n].copy()
                    if n in ("kv", "index", "repair")
                    else actual[n][i : i + 1].copy()
                )
                for n in FIELDS
            }
        )
    rows[0]["kv"][:] = 1
    result = stack_reference(rows)
    assert all(equal_bytes(result[n], actual[n]) for n in FIELDS)
    with pytest.raises(ValueError):
        stack_reference([])


def test_local_observer_never_converts_global_array():
    actual, _ = observations(host("boundary"), slot=29, layer=0)

    class Distributed:
        def __init__(self, value):
            self.addressable_shards = [
                SimpleNamespace(device=SimpleNamespace(id=99), data=value)
            ]

        def __array__(self, *args, **kwargs):
            raise AssertionError("global transfer is forbidden")

    values = []
    for name in FIELDS:
        v = actual[name]
        if name in ("kv", "index", "repair"):
            v = v[None]
        elif name == "health":
            v = v[None, None]
        values.append(Distributed(v))
    result = local_observations(tuple(values))[99]
    assert all(equal_bytes(result[n], actual[n]) for n in FIELDS)


@pytest.mark.parametrize(
    "bad",
    [
        "greenfield_fp8_ws32_prefill_moe_admission_x",
        "greenfield_fp8_ws32_prefill_layer_admission_l1_x",
        "../x",
        "",
    ],
)
def test_layer_tag_refuses_cross_protocol_and_unregistered_layer(bad):
    with pytest.raises(ValueError):
        layer_from_tag(bad)


def test_bf16_archive_cannot_use_wrong_storage_dtype():
    arrays = encode_arrays("x", {"kv": np.zeros((2, 64, 640), BF16)})
    arrays["x__kv"] = arrays["x__kv"].astype(np.float32)
    with pytest.raises(ValueError, match="storage dtype"):
        decode_arrays(arrays, "x", ("kv",))
