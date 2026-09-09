"""Original-array panel admission without model execution or cloud access."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_phase_originals as old
from scripts.greenfield import prefill_panel_originals as panel
from scripts.greenfield import prefill_completed_window_protocol as completed
from scripts.greenfield import prefill_phase_variant as variants
from scripts.greenfield import prefill_phase_evidence as evidence
from scripts.greenfield.prefill_layer_evidence import INPUT_FIELDS, decode_arrays


@pytest.fixture(scope="module")
def originals():
    capsule = old.load_capsule()
    source = Path("/home/gianl/glm-run") / capsule["source_tag"] / "fleet/rank0"
    # Mandatory for this local admission; never silently skip missing evidence.
    assert source.is_dir()
    slots = {
        o["device_id"]: int(s)
        for s, o in capsule["owners"].items()
        if o["launch_rank"] == 0
    }
    record = dict(
        physical_device_ids=capsule["physical_device_ids"],
        mesh_sha256=capsule["mesh_sha256"],
        local_device_slots=[
            dict(
                device_id=d,
                device_slot=s,
                observed_selected_tensor_sha256=capsule["owners"][str(s)][
                    "selected_tensor_sha256"
                ],
            )
            for d, s in slots.items()
        ],
    )
    with np.load(source / "competitive.npz", allow_pickle=False) as arrays:
        raw = {k: arrays[k] for k in arrays.files}
    host = decode_arrays(raw, "input", INPUT_FIELDS)
    return capsule, slots, record, raw, host, source


def observations(raw, slots, kind):
    return {
        d: completed._decode(raw, f"{kind}_{d}", old.COMPONENTS[kind]) for d in slots
    }


@pytest.mark.parametrize(
    "change", ["bounded", "too_large", "nonfinite", "routes", "repeat", "assembly"]
)
def test_first_original_control_bounds_before_repeats(tmp_path, originals, change):
    capsule, slots, record, raw, host, source = originals
    calls = SimpleNamespace(root=tmp_path, record=deepcopy(record), local_slots=slots)
    verifier = panel.PanelOriginalVerifier(calls, capsule, host)
    d = next(iter(slots))
    data = {k: observations(raw, slots, k) for k in old.COMPONENTS}
    y = data["wide"][d]["output"].copy()
    y[0, 0] = np.asarray(
        float(y[0, 0]) + (0.5 if change == "too_large" else 0.0078125), dtype=y.dtype
    )
    if change == "nonfinite":
        y[0, 0] = np.nan
    data["wide"][d]["output"] = y
    data["actual"][d]["output"] = y.copy()
    if change == "routes":
        data["wide"][d]["routes"] = data["wide"][d]["routes"].copy()
        data["wide"][d]["routes"][0, 0] = -1
    if change == "assembly":
        data["actual"][d]["output"] = data["actual"][d]["output"].copy() + np.asarray(
            0.25, dtype=y.dtype
        )

    def traverse():
        for kind in old.COMPONENTS:
            verifier.capture(kind, data[kind])

    if change in ("too_large", "nonfinite", "routes", "assembly"):
        with pytest.raises(ValueError):
            traverse()
        assert (tmp_path / "phase_first.npz").exists()
        assert not verifier.report["complete"]
        return
    traverse()
    assert verifier.report["panel_bounded_reference"]["passed"]
    if change == "repeat":
        data["wide"][d]["output"] = y.copy() + np.asarray(0.00390625, dtype=y.dtype)
        with pytest.raises(ValueError, match="repeat"):
            traverse()
        assert (tmp_path / "phase_failure.npz").exists()
    else:
        traverse()
        verifier.finish(2)
        assert verifier.report["complete"]
        assert (
            panel.bounded_receipt(tmp_path / "phase_first.npz", slots)
            == verifier.report["panel_bounded_reference"]
        )
        # Nonzero admissible output change is still refused by historical mode.
        with pytest.raises(ValueError):
            old.check_observation(
                capsule, slot=slots[d], kind="wide", values=data["wide"][d]
            )


def test_collector_recomputes_bounds_and_refuses_fitted_worker_report(
    tmp_path, originals, monkeypatch
):
    import os

    capsule, slots, record, raw, host, source = originals
    variant = variants.variants()[2]
    record = deepcopy(record)
    record.update(
        protocol=variant.protocol,
        profile=variant.admission.PROFILE,
        original_binding=old.bind_originals(record, slots, capsule),
    )
    verifier = panel.PanelOriginalVerifier(
        SimpleNamespace(root=tmp_path, record=record, local_slots=slots), capsule, host
    )
    for kind in old.COMPONENTS:
        verifier.capture(kind, observations(raw, slots, kind))
    verifier.finish(1)
    record["original_authentication"]["visits"] = {k: 15 for k in old.COMPONENTS}
    for kind in ("wk_decode", "wk_promote"):
        os.link(source / f"{kind}.npz", tmp_path / f"{kind}.npz")
    # Existing complete WK validation is exercised by the composed campaign;
    # this local test separately retains exact original WK byte checks.
    monkeypatch.setattr(evidence.shared, "validate_wk", lambda *a: None)
    evidence.validate_originals(tmp_path, record, slots)
    receipt = record["original_authentication"]["panel_bounded_reference"]
    import json
    from hashlib import sha256

    full = panel.bounded_replay(tmp_path / "phase_first.npz", slots)
    canonical = json.dumps(
        full, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert receipt["report_sha256"] == sha256(canonical).hexdigest()
    assert receipt["report_bytes"] == len(canonical)
    assert len(json.dumps(receipt)) < 256
    assert len(canonical) > 10000  # full details are rederived, never discarded
    for key, bad in (
        ("passed", False),
        ("report_sha256", "0" * 64),
        ("report_bytes", 1),
    ):
        before = receipt[key]
        receipt[key] = bad
        with pytest.raises(ValueError):
            evidence.validate_originals(tmp_path, record, slots)
        receipt[key] = before
