"""Actual retained long graph checks, plus early provenance refusals."""

from hashlib import sha256
from pathlib import Path
import json

import pytest

from scripts.greenfield import ws32_delivery_hlo as delivery

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/home/gianl/glm-run")
L7 = BASE / "greenfield_fp8_ws32_delivery_long_prefill_compile_20260911T232546214124179Z/rank0"
E0 = BASE / "greenfield_fp8_ws32_capture_barrier_prefill_compile_20260912T025701169016006Z/fleet/rank0"
CASES = (
    ("128k_d1_0", "prefill_chunk", L7 / "prefill_128k_chunk", "ff0c56c1e8c1dde0112592193fd77e3cde053d7859647fc3ab9777d634da20b2"),
    ("128k_d1_0", "prefill_tail", L7 / "prefill_128k_tail", "ecd21afae7c5ea0d16e6bbbd77d4600b39d61018d4d321f779d88cae2c2d69be"),
    ("256k_e0", "prefill_chunk", E0 / "prefill_256k_capture_barrier", "c11cd29d33f9750b9e0bc81ff17c6e6df88d28d988feff398481e0f448096099"),
)


@pytest.mark.parametrize("label,role,path,digest", CASES)
def test_complete_actual_long_graph_report(label, role, path, digest):
    # Missing evidence is a failure, not an unnoticed skip of the new path.
    stable = Path(str(path) + ".stablehlo.mlir").read_text()
    optimized = Path(str(path) + ".optimized_hlo.txt").read_text()
    report = delivery.inspect_hlo(stable, optimized, repo=ROOT, context_label=label,
                                  role=role, expected_stablehlo_sha256=sha256(stable.encode()).hexdigest(),
                                  expected_optimized_sha256=digest)
    assert report["passed"], report
    assert report["dispatch_authorized"] is False
    assert report["numerical_claim"] is report["performance_claim"] is False
    checks = report["checks"]
    assert report["maximum_group_size"] == 8
    assert checks["collectives"]["static_collective_count"] == 788
    assert checks["kernels"]["kernel_count"] == 1047
    assert checks["kernels"]["family_counts"] == {"raw": 588, "panels": 225, "structured": 156, "sparse": 78}
    assert len(checks["fp32_moe_route_sums"]["layers"]) == 75
    assert checks["large_cache_storage"]["instruction_count"] == (23 if label == "256k_e0" else 328)
    assert json.loads(json.dumps(report)) == report
    print(label, role, checks["large_cache_storage"]["instruction_count"], digest, flush=True)


@pytest.mark.parametrize("case", ["workload", "role", "stable_pin", "digest", "raw", "optimized", "source"])
def test_early_provenance_refusals(monkeypatch, case):
    monkeypatch.setattr(delivery, "parse_hlo_module", lambda _: pytest.fail("parsed refused bytes"))
    kwargs = dict(repo=ROOT, context_label="256k_e0", role="prefill_chunk",
                  expected_stablehlo_sha256=delivery.programs.raw_registration("256k_e0")["prefill_chunk"][1],
                  expected_optimized_sha256="0" * 64)
    if case == "workload": kwargs["context_label"] = "8k"
    elif case == "role": kwargs["role"] = "decode"
    elif case == "digest": kwargs["expected_optimized_sha256"] = None
    elif case == "stable_pin": kwargs["expected_stablehlo_sha256"] = "0" * 64
    elif case == "source":
        def fail(_): raise ValueError("source changed")
        monkeypatch.setattr(delivery.programs, "require_source", fail)
    elif case == "optimized":
        monkeypatch.setattr(delivery.programs, "raw_registration", lambda _: {"prefill_chunk": (0, sha256(b"").hexdigest())})
        monkeypatch.setattr(delivery.programs, "require_source", lambda _: None)
        kwargs["expected_stablehlo_sha256"] = sha256(b"").hexdigest()
    with pytest.raises(ValueError):
        delivery.inspect_hlo("", "", **kwargs)


@pytest.mark.parametrize("capacity,rows", [(8192,128), (262656,114), (131072,32), (131072,128.0)])
def test_long_geometry_refuses_before_parse(capacity, rows):
    with pytest.raises(ValueError, match="geometry"):
        delivery.check_index(None, context_capacity=capacity, block_rows=rows, live_instructions=())
