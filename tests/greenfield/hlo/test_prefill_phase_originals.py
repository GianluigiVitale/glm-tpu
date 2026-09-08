"""Compact original provenance and strict field/type/owner reproduction."""

from hashlib import sha256
import io
from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield import prefill_phase_originals as original


def test_committed_capsule_binds_all_owners_components_and_source_generations():
    capsule = original.load_capsule()
    assert capsule["source_db"] == 594 and capsule["case"] == "competitive"
    assert len(capsule["sources"]) == 32  # runner + case + two WK, eight hosts
    assert len({r["name"] for r in capsule["sources"]}) == 32
    assert all(
        r["generation"].isdigit() and r["crc32c"] and r["size"] > 0
        for r in capsule["sources"]
    )
    assert set(capsule["owners"]) == {str(s) for s in range(32)}
    order = [d for group in capsule["physical_device_ids"] for d in group]
    assert sorted(order) == list(range(32))
    for slot, owner in capsule["owners"].items():
        assert owner["device_id"] == order[int(slot)]
        assert len(owner["selected_tensor_sha256"]) == 35
        assert set(owner["components"]) == set(original.COMPONENTS)
        for kind, fields in original.COMPONENTS.items():
            assert set(owner["components"][kind]) == set(fields)


@pytest.mark.parametrize(
    "change", ["none", "mesh", "slot", "weights", "duplicate", "float"]
)
def test_original_runtime_binding_checks_typed_mesh_and_selected_weights(change):
    capsule = original.load_capsule()
    owners = [o for o in capsule["owners"].values() if o["launch_rank"] == 0]
    slots = {
        o["device_id"]: s
        for s in range(32)
        for o in [capsule["owners"][str(s)]]
        if o["launch_rank"] == 0
    }
    record = dict(
        physical_device_ids=tuple(tuple(g) for g in capsule["physical_device_ids"]),
        mesh_sha256=capsule["mesh_sha256"],
        local_device_slots=[
            dict(
                device_id=o["device_id"],
                device_slot=slots[o["device_id"]],
                observed_selected_tensor_sha256=dict(o["selected_tensor_sha256"]),
            )
            for o in owners
        ],
    )
    if change == "mesh":
        record["physical_device_ids"] = tuple(reversed(record["physical_device_ids"]))
    elif change == "slot":
        d = next(iter(slots))
        slots[d] = (slots[d] + 1) % 32
    elif change == "weights":
        record["local_device_slots"][0]["observed_selected_tensor_sha256"].clear()
    elif change == "duplicate":
        record["local_device_slots"][0] = record["local_device_slots"][1]
    elif change == "float":
        d = next(iter(slots))
        slots[d] = float(slots[d])
    if change == "none":
        assert original.bind_originals(record, slots, capsule)["source_db"] == 594
    else:
        with pytest.raises(ValueError):
            original.bind_originals(record, slots, capsule)


@pytest.mark.parametrize(
    "change", ["none", "value", "dtype", "shape", "missing", "extra"]
)
def test_observation_requires_all_fields_shape_dtype_and_bytes(change):
    values = dict(output=np.arange(4, dtype=np.uint16).reshape(2, 2))
    capsule = dict(owners={"0": dict(components={"wide": original.manifest(values)})})
    if change == "value":
        values["output"][0, 0] = 7
    elif change == "dtype":
        values["output"] = values["output"].view(np.int16)
    elif change == "shape":
        values["output"] = values["output"].reshape(4)
    elif change == "missing":
        values.clear()
    elif change == "extra":
        values["extra"] = np.zeros(1)
    if change == "none":
        assert original.check_observation(capsule, slot=0, kind="wide", values=values)
    else:
        with pytest.raises(ValueError, match="mismatch"):
            original.check_observation(capsule, slot=0, kind="wide", values=values)


def test_capsule_tamper_refused(tmp_path, monkeypatch):
    path = tmp_path / "changed.json"
    path.write_bytes(original.CAPSULE.read_bytes() + b" ")
    monkeypatch.setattr(original, "CAPSULE", path)
    with pytest.raises(ValueError, match="digest"):
        original.load_capsule()


def test_local_archived_db594_outputs_reproduce_capsule():
    # Optional real-data regression; no TPU initialization, numerics or cloud IO.
    capsule = original.load_capsule()
    root = Path("/home/gianl/glm-run") / capsule["source_tag"]
    if not root.exists():
        pytest.skip("archived original run not local")
    for rank in range(8):
        relative = f"fleet/rank{rank}/competitive.npz"
        source = next(
            r for r in capsule["sources"] if r["name"].endswith("/" + relative)
        )
        raw = (root / relative).read_bytes()
        assert (
            len(raw) == source["size"]
            and sha256(raw).hexdigest() == source["original_sha256"]
        )
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            for s, owner in capsule["owners"].items():
                if owner["launch_rank"] != rank:
                    continue
                for kind, fields in original.COMPONENTS.items():
                    values = {
                        n: arrays[f"{kind}_{owner['device_id']}__{n}"] for n in fields
                    }
                    original.check_observation(
                        capsule, slot=int(s), kind=kind, values=values
                    )


@pytest.mark.parametrize("failure", [None, "first", "later"])
def test_verifier_retains_one_capture_and_preserves_mismatch(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    monkeypatch.setattr(original, "bind_originals", lambda *a: {"test_binding": True})
    fields = {"output": np.asarray([1.0], np.float32)}
    inputs = {"update": np.ones(1, np.float32)}
    source = dict(
        inputs=original.manifest(inputs),
        owners={
            "0": dict(
                components={k: original.manifest(fields) for k in original.COMPONENTS}
            )
        },
    )
    calls = SimpleNamespace(record={}, local_slots={8: 0}, root=tmp_path)
    verifier = original.OriginalVerifier(calls, source, inputs)
    for iteration in range(2):
        for kind in original.COMPONENTS:
            if kind == "prefix0" and (
                (failure == "first" and iteration == 0)
                or (failure == "later" and iteration == 1)
            ):
                with pytest.raises(ValueError, match="mismatch"):
                    verifier.capture(
                        kind, {8: {"output": np.asarray([2.0], np.float32)}}
                    )
                name = "phase_first.npz" if failure == "first" else "phase_failure.npz"
                with np.load(tmp_path / name) as arrays:
                    assert arrays["prefix0_8__output"][0] == 2.0
                assert not verifier.report["complete"]
                return
            verifier.capture(kind, {8: fields})
        assert not verifier.arrays
        first_bytes = (tmp_path / "phase_first.npz").read_bytes()
        assert sha256(first_bytes).hexdigest() == verifier.report["first_npz_sha256"]
    verifier.finish(2)
    assert verifier.report["complete"] and not (tmp_path / "phase_failure.npz").exists()
