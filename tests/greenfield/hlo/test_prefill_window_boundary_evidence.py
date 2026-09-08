"""Actual diagnostic producer/JSON/NPZ/journal consumer; model math is mocked."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from scripts.greenfield import prefill_window_boundary_evidence as evidence
from scripts.greenfield import prefill_window_boundary_worker as boundary
from tests.greenfield.hlo.test_prefill_window_evidence import completed_worker


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    with completed_worker(
        tmp_path_factory.mktemp("boundary-composed"), boundary_diagnostic=True
    ) as result:
        yield result


def test_actual_compiler_worker_json_journal_npz_replay(completed):
    root, record = completed
    evidence.validate_files(root, record)
    assert record["model_executable_calls"] == 5 and record["wk_executable_calls"] == 2
    assert record["boundary_capture_complete"]
    assert not record[
        "original_signature_reproduced"
    ]  # mocked outputs, no original math
    assert record["classification"] == "INSTRUMENTATION_PERTURBED_ORIGINAL_SIGNATURE"


@pytest.mark.parametrize(
    "change",
    [
        "count",
        "order",
        "memory",
        "scope",
        "manifest",
        "fingerprint",
        "binding",
        "outcome",
        "schema",
        "journal",
    ],
)
def test_declared_evidence_cannot_replace_actuals(completed, change):
    root, source = completed
    record = deepcopy(source)
    if change == "count":
        record["model_executable_calls"] = 15
    elif change == "order":
        record["call_evidence"][3]["phase"] = "boundary/control1"
    elif change == "memory":
        record["call_evidence"][2]["budget"]["devices"][0]["estimated_peak_bytes"] += 1
    elif change == "scope":
        record["diagnostic_only"] = False
    elif change == "manifest":
        record["captures"]["actual"]["0"]["router/input"]["sha256"] = "0" * 64
    elif change == "fingerprint":
        record["original_reproduction"]["0"]["actual"]["all_outputs_reproduced"] = True
    elif change == "binding":
        record["original_binding"]["sources"][0]["ledger"]["generation"] = "1"
    elif change == "outcome":
        record["original_signature_reproduced"] = True
    elif change == "schema":
        record["programs"]["candidate"]["compiler_output_schema"]["captures"][
            "router/input"
        ]["dtype"] = "float32"
    else:
        record["compile_journal_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        evidence.validate_files(root, record)


@pytest.mark.parametrize(
    "change", ["nan", "dtype", "shape", "extra", "missing", "control"]
)
def test_rehashed_array_mutations_still_refuse(completed, tmp_path, change):
    root, source = completed
    record = deepcopy(source)
    with np.load(root / "boundary.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    key = "capture_actual_0__router/input"
    if change == "nan":
        arrays[key].view(boundary.window.BF16)[0, 0] = np.nan
    elif change == "dtype":
        arrays[key] = arrays[key].astype(np.float32)
    elif change == "shape":
        arrays[key] = arrays[key].reshape(-1)
    elif change == "extra":
        arrays["unregistered"] = np.zeros(1)
    elif change == "missing":
        del arrays[key]
    else:
        arrays["control_0__output"][0, 0] ^= 1
    with (tmp_path / "boundary.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    record["boundary_npz_sha256"] = sha256(
        (tmp_path / "boundary.npz").read_bytes()
    ).hexdigest()
    slots = {s["device_id"]: s["device_slot"] for s in record["local_device_slots"]}
    with pytest.raises(KeyError if change == "missing" else ValueError):
        evidence.replay_arrays(tmp_path, record, slots)
