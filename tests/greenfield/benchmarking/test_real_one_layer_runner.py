from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "greenfield"
    / "run_real_one_layer.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_real_one_layer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_record_normalizes_numpy_scalars_and_paths(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "record.json"
    module._atomic_write(
        output,
        {
            "device_id": np.int64(7),
            "latency": np.float64(0.625),
            "path": tmp_path,
        },
    )
    assert json.loads(output.read_text()) == {
        "device_id": 7,
        "latency": 0.625,
        "path": str(tmp_path),
    }


def test_pallas_stage_adapter_reorders_router_and_correction_bias(
    monkeypatch,
) -> None:
    module = _load_script()
    values = tuple(object() for _ in range(15))
    local_expert_shard = object()
    contract = object()
    observed = {}

    def fake_pallas_stage(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "mapped"

    monkeypatch.setattr(
        module, "stage_local_moe_pallas_mapped", fake_pallas_stage
    )
    result = module._pallas_stage_step(
        *values,
        local_expert_shard,
        axis_name="expert",
        contract=contract,
    )

    assert result == "mapped"
    assert observed["args"] == (
        values[0],
        values[2],
        values[1],
        *values[3:],
        local_expert_shard,
    )
    assert observed["kwargs"] == {
        "axis_name": "expert",
        "contract": contract,
    }
