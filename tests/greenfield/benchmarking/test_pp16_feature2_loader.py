from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import (
    _read_manifest,
    _read_authenticated_range,
    inspect_feature2_selective_plan,
)
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
    Feature2TensorRead,
)
from glm_tpu.greenfield.errors import CheckpointValidationError


REAL_PP16_FEATURE2_RUNTIME = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_"
    "20260827T164842844148623Z"
)


def _tensor(
    *,
    slot: int,
    filename: str,
    name: str,
    offset: int,
    value: bytes,
    dtype: str,
    shape: tuple[int, ...],
) -> Feature2TensorRead:
    return Feature2TensorRead(
        device_slot=slot,
        filename=filename,
        name=name,
        offset=offset,
        byte_count=len(value),
        dtype=dtype,
        shape=shape,
        sha256=sha256(value).hexdigest(),
    )


@pytest.mark.skipif(
    not REAL_PP16_FEATURE2_RUNTIME.is_dir(),
    reason="protected PP16 feature runtime is unavailable",
)
def test_real_feature2_selective_plan_is_exact_and_payload_free() -> None:
    report = inspect_feature2_selective_plan(REAL_PP16_FEATURE2_RUNTIME)
    assert report == {
        "bytes_by_slot": {0: 1_199_760_512, 1: 1_199_760_512},
        "manifest_sha256": PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
        "read_count": 78,
        "tensor_names": report["tensor_names"],
    }
    assert len(report["tensor_names"]) == 39
    assert "global.embedding" in report["tensor_names"]
    assert "dense.slot_01.down.weight_bits" not in report["tensor_names"]


@pytest.mark.skipif(
    not REAL_PP16_FEATURE2_RUNTIME.is_dir(),
    reason="protected PP16 feature runtime is unavailable",
)
def test_manifest_is_authenticated_before_manifest_directed_io(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (REAL_PP16_FEATURE2_RUNTIME / "runtime_manifest.json").read_text()
    )
    manifest["files"][0]["destination_filename"] = "../../untrusted"
    (tmp_path / "runtime_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "SUCCESS").write_text(
        f"{PP16_FEATURE2_RUNTIME_MANIFEST_SHA256}  runtime_manifest.json\n"
    )
    with pytest.raises(
        CheckpointValidationError, match="manifest authentication failed"
    ):
        _read_manifest(tmp_path)


def test_range_read_refuses_sha_shape_truncation_and_nonfinite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "owner.bin"
    finite = np.asarray([1.0, 2.0], dtype=np.float32).tobytes()
    path.write_bytes(b"prefix00" + finite)
    tensor = _tensor(
        slot=0,
        filename=path.name,
        name="finite",
        offset=8,
        value=finite,
        dtype="F32",
        shape=(2,),
    )
    host, observed = _read_authenticated_range(path, tensor, chunk_bytes=5)
    assert observed == tensor.sha256
    path.write_bytes(b"prefix00" + np.asarray([9.0, 9.0], np.float32).tobytes())
    assert np.array_equal(host, np.asarray([[1.0, 2.0]], np.float32))
    with pytest.raises(CheckpointValidationError, match="SHA-256"):
        _read_authenticated_range(
            path,
            Feature2TensorRead(
                **{**tensor.to_dict(), "sha256": "0" * 64}
            ),
            chunk_bytes=4,
        )
    with pytest.raises(CheckpointValidationError, match="shape bytes"):
        _read_authenticated_range(
            path,
            Feature2TensorRead(
                **{**tensor.to_dict(), "shape": (1,)}
            ),
            chunk_bytes=4,
        )
    with pytest.raises(CheckpointValidationError, match="truncated"):
        _read_authenticated_range(
            path,
            Feature2TensorRead(
                **{
                    **tensor.to_dict(),
                    "byte_count": 12,
                    "shape": (3,),
                    "sha256": "0" * 64,
                }
            ),
            chunk_bytes=4,
        )
    nonfinite = np.asarray([np.inf], dtype=np.float32).tobytes()
    path.write_bytes(b"prefix00" + nonfinite)
    with pytest.raises(CheckpointValidationError, match="non-finite"):
        _read_authenticated_range(
            path,
            _tensor(
                slot=0,
                filename=path.name,
                name="nonfinite",
                offset=8,
                value=nonfinite,
                dtype="F32",
                shape=(1,),
            ),
            chunk_bytes=4,
        )
    fp8_nan = bytes([0x7F])
    path.write_bytes(b"prefix00" + fp8_nan)
    with pytest.raises(CheckpointValidationError, match="non-finite"):
        _read_authenticated_range(
            path,
            _tensor(
                slot=0,
                filename=path.name,
                name="fp8.weight_bits",
                offset=8,
                value=fp8_nan,
                dtype="U8",
                shape=(1,),
            ),
            chunk_bytes=1,
        )


def test_forced_two_loader_places_only_exact_ranges_on_final_owners(
    tmp_path: Path,
) -> None:
    payloads: dict[int, list[tuple[str, str, tuple[int, ...], bytes]]] = {}
    for slot in (0, 1):
        payloads[slot] = [
            (
                "bits.weight_bits",
                "U8",
                (4,),
                bytes([slot + 1, 2, 3, 4]),
            ),
            (
                "norm",
                "BF16",
                (2,),
                np.asarray(
                    [slot + 0.5, slot + 1.5], dtype=ml_dtypes.bfloat16
                ).tobytes(),
            ),
            (
                "scale",
                "F32",
                (1,),
                np.asarray([slot + 2.25], dtype=np.float32).tobytes(),
            ),
        ]
    allowed = []
    for slot, values in payloads.items():
        filename = f"owner{slot}.bin"
        raw = bytearray(b"prefix00")
        for name, dtype, shape, value in values:
            offset = len(raw)
            raw.extend(value)
            allowed.append(
                _tensor(
                    slot=slot,
                    filename=filename,
                    name=name,
                    offset=offset,
                    value=value,
                    dtype=dtype,
                    shape=shape,
                )
            )
        (tmp_path / filename).write_bytes(raw)

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps([tensor.to_dict() for tensor in allowed], sort_keys=True)
    )
    program = r'''
import json
from pathlib import Path
import jax
import numpy as np
from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import (
    _load_ranges_to_final_owners,
)
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    Feature2TensorRead,
)

root = Path(__import__("os").environ["FEATURE2_TEST_ROOT"])
allowed = tuple(
    Feature2TensorRead(**item)
    for item in json.loads((root / "plan.json").read_text())
)
weights, receipts, placement = _load_ranges_to_final_owners(
    root,
    allowed,
    jax.devices(),
    axis_name="feature",
    chunk_bytes=5,
)
result = {
    "bits": np.asarray(weights["bits.weight_bits"]).tolist(),
    "norm_shape": list(weights["norm"].shape),
    "placement": placement,
    "receipt_count": len(receipts),
    "scale": np.asarray(weights["scale"]).tolist(),
}
for value in weights.values():
    value.delete()
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ)
    env["FEATURE2_TEST_ROOT"] = str(tmp_path)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["bits"] == [[1, 2, 3, 4], [2, 2, 3, 4]]
    assert result["norm_shape"] == [2, 2]
    assert result["placement"] == {
        "device_roundtrip_bytes": 24,
        "mesh_axis": "feature",
        "owner_device_ids": [0, 1],
    }
    assert result["receipt_count"] == 6
    assert result["scale"] == [[2.25], [3.25]]
