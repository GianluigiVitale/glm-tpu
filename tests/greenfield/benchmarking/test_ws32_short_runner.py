from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/run_short_decoder_ws32.py"
SPEC = importlib.util.spec_from_file_location("ws32_short_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_ws32_short_runner_latency_distribution_is_exact() -> None:
    result = RUNNER._distribution([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "count": 4,
        "maximum_ms": 4.0,
        "mean_ms": 2.5,
        "minimum_ms": 1.0,
        "p50_ms": 2.5,
        "p90_ms": pytest.approx(3.7),
        "p95_ms": pytest.approx(3.85),
        "p99_ms": pytest.approx(3.97),
    }
    with pytest.raises(ValueError, match="requires samples"):
        RUNNER._distribution([])


def test_ws32_short_runner_evidence_is_append_only_and_hashed(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.json"
    RUNNER._atomic_text(record, "one\n")
    with pytest.raises(FileExistsError, match="append-only"):
        RUNNER._atomic_text(record, "two\n")
    trace = tmp_path / "trace"
    payload = trace / "plugins/profile/run/trace.xplane.pb"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"trace")
    assert RUNNER._trace_files(trace) == [
        {
            "byte_count": 5,
            "relative_path": "plugins/profile/run/trace.xplane.pb",
            "sha256": sha256(b"trace").hexdigest(),
        }
    ]
    with pytest.raises(RuntimeError, match="0 XPlane files"):
        RUNNER._trace_files(tmp_path / "empty")
    tensor_path = tmp_path / "runner.rank0.npz"
    tensor_record = RUNNER._atomic_npz(
        tensor_path,
        positions=np.asarray([1, 2], dtype=np.int32),
    )
    assert tensor_record["filename"] == "runner.rank0.npz"
    assert tensor_record["arrays"]["positions"]["shape"] == [2]
    assert tensor_record["arrays"]["positions"]["dtype"] == "int32"
    with pytest.raises(FileExistsError, match="append-only"):
        RUNNER._atomic_npz(
            tensor_path,
            positions=np.asarray([3], dtype=np.int32),
        )


def test_ws32_short_runner_is_default_off_and_independent() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--compile-only" in source
    assert "verify_ws32_runtime_checkpoint(" in source
    assert "compare_ws32_dsa_step(" in source
    assert "compare_ws32_raw_tokens(" in source
    assert "validate_ws32_cache_probe(" in source
    assert "jax.profiler.trace(" in source
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION") != _XLA_MEMORY_FRACTION' in source
    assert '"token_oracle_success_sha256"' in source
    assert '"dsa_oracle_success_sha256"' in source
    assert "performance_claim\": False" in source
    assert "tpu_inference" not in source
    assert "vllm" not in source
