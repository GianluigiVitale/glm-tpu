"""Production packet geometry through real addressable-only readers; CPU arrays."""

from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_norm_worker as worker

SLOTS = {9: 9, 25: 25, 13: 13, 29: 29}


class Owned:
    def __init__(self, shape, dtype, fill):
        self.shape = (8, 4, *shape)
        self.dtype = np.dtype(dtype)
        self.addressable_shards = []
        for device, slot in SLOTS.items():
            expert, feature = divmod(slot, 4)
            data = np.full((1, 1, *shape), fill, dtype)
            self.addressable_shards.append(
                NS(
                    device=NS(id=device),
                    data=data,
                    index=(
                        slice(expert, expert + 1),
                        slice(feature, feature + 1),
                        *(slice(None) for _ in shape),
                    ),
                )
            )

    def __array__(self, *args, **kwargs):
        raise AssertionError("global host reconstruction forbidden")


def packet(count):
    fields = {
        f"post_norm/{n}": Owned((128, 1536), ml_dtypes.bfloat16, 1)
        for n in ("update", "residual", "normalized", "carried")
    }
    fields["post_norm/summed"] = Owned((128, 1536), np.float32, 2)
    for name in ("local_square_sum", "square_sum", "inverse"):
        fields[f"post_norm/{name}"] = Owned((128, 1), np.float32, 1)
    fields["post_norm/weight"] = Owned((1536,), ml_dtypes.bfloat16, 1)
    fields["boundary/normalized_mlp"] = Owned((128, 1536), ml_dtypes.bfloat16, 1)
    fields["boundary/live"] = Owned((128,), np.bool_, False)
    for shard in fields["boundary/normalized_mlp"].addressable_shards:
        shard.data[:, :, count:] = 0
    for shard in fields["boundary/live"].addressable_shards:
        shard.data[:, :, :count] = True
    return fields


@pytest.mark.parametrize("count,expected", [(128, 11028992), (32, 2766848)])
def test_live_only_packet_full_mask_and_actual_nonidentity_owner_slots(count, expected):
    arrays, report = worker.capture_packet(packet(count), slots=SLOTS, count=count)
    assert report["valid"] and len(arrays) == 44
    assert sum(a.nbytes for a in arrays.values()) == expected
    for slot in SLOTS.values():
        assert arrays[f"slot{slot}__post_norm_weight"].shape == (1536,)
        assert arrays[f"slot{slot}__post_norm_summed"].shape == (count, 1536)
        assert arrays[f"slot{slot}__post_norm_normalized"].dtype == np.uint16
        np.testing.assert_array_equal(
            arrays[f"slot{slot}__boundary_live"], np.arange(128) < count
        )


@pytest.mark.parametrize(
    "field", ["post_norm/summed", "post_norm/weight", "boundary/normalized_mlp"]
)
def test_nonfinite_live_capture_is_returned_for_preservation(field):
    fields = packet(32)
    fields[field].addressable_shards[0].data.flat[0] = np.nan
    arrays, report = worker.capture_packet(fields, slots=SLOTS, count=32)
    assert (
        arrays
        and not report["valid"]
        and any("nonfinite" in e for e in report["errors"])
    )


@pytest.mark.parametrize(
    "change", ["mask", "padding", "missing", "owner", "duplicate", "shape"]
)
def test_malformed_packet_refuses_or_preserves_invalid_content(change):
    fields = packet(32)
    if change == "mask":
        fields["boundary/live"].addressable_shards[0].data.flat[90] = True
    elif change == "padding":
        fields["boundary/normalized_mlp"].addressable_shards[0].data[0, 0, 50, 0] = 1
    elif change == "missing":
        fields.pop("post_norm/weight")
    elif change == "owner":
        fields["post_norm/weight"].addressable_shards[0].index = (
            slice(0, 1),
            slice(0, 1),
        )
    elif change == "duplicate":
        fields["post_norm/weight"].addressable_shards.append(
            fields["post_norm/weight"].addressable_shards[0]
        )
    else:
        fields["post_norm/weight"].shape = (8, 4, 1, 1536)
    if change in ("mask", "padding"):
        arrays, report = worker.capture_packet(fields, slots=SLOTS, count=32)
        assert arrays and not report["valid"]
    else:
        with pytest.raises(ValueError):
            worker.capture_packet(fields, slots=SLOTS, count=32)


@pytest.mark.parametrize("fault", [None, "health", "padding", "nonfinite"])
def test_suffix_full_health_and_padding_check_before_trim(fault):
    output = Owned((128, 1536), ml_dtypes.bfloat16, 0)
    health = Owned((128,), np.bool_, True)
    if fault == "health":
        health.addressable_shards[0].data.flat[127] = False
    elif fault == "padding":
        output.addressable_shards[0].data[0, 0, 127, 0] = 1
    elif fault == "nonfinite":
        output.addressable_shards[0].data.flat[0] = np.nan
    arrays, report = worker.capture_suffix((output, health), slots=SLOTS, count=32)
    assert report["valid"] == (fault is None)
    assert all(arrays[f"slot{s}__health"].shape == (128,) for s in SLOTS.values())
    assert all(arrays[f"slot{s}__output"].shape == (32, 1536) for s in SLOTS.values())
