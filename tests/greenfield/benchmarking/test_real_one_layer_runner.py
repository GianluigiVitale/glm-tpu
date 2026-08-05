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
