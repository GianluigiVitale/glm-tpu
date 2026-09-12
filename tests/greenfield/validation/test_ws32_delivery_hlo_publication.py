"""Long evidence writer/JSON/sealer composition; no TPU compilation or call."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_delivery_hlo as delivery

ROOT = Path(__file__).resolve().parents[3]


def values(tmp_path, stable="untrusted raw", optimized="untrusted optimized"):
    return dict(
        graph="prefill_chunk",
        lowered=SimpleNamespace(compiler_ir=lambda **_: stable),
        compiled=SimpleNamespace(as_text=lambda: optimized),
        hlo_dir=tmp_path, expected_stable=sha256(stable.encode()).hexdigest(),
        expected_optimized=sha256(optimized.encode()).hexdigest(),
        hidden_size=6144, exact_dsa=True, strategy_nd_dense=True,
        host_main_rope_table=True, prefill_mode=worker.PREFILL_MODE,
        block_rows=128, batched_profile=delivery.PROFILE,
        long_context_label="256k_e0",
    )


def test_actual_e0_writer_json_and_sealer_replay(tmp_path):
    directory = Path("/home/gianl/glm-run/greenfield_fp8_ws32_capture_barrier_prefill_compile_20260912T025701169016006Z/fleet/rank0")
    stable = (directory / "prefill_256k_capture_barrier.stablehlo.mlir").read_text()
    optimized = (directory / "prefill_256k_capture_barrier.optimized_hlo.txt").read_text()
    assert sha256(optimized.encode()).hexdigest() == "c11cd29d33f9750b9e0bc81ff17c6e6df88d28d988feff398481e0f448096099"
    kwargs = values(tmp_path, stable, optimized)
    report, actual_stable, actual_optimized = worker._write_graph(**kwargs)
    assert (actual_stable, actual_optimized) == (stable, optimized)
    assert (tmp_path / "prefill_chunk.stablehlo.mlir").read_text() == stable
    assert (tmp_path / "prefill_chunk.optimized_hlo.txt").read_text() == optimized
    worker._atomic_json(tmp_path / "graph.json", report)
    serialized = json.loads((tmp_path / "graph.json").read_text())
    args = SimpleNamespace(
        batched_prefill_profile=delivery.PROFILE, context_label="256k_e0",
        expected_prefill_chunk_stablehlo_sha256=kwargs["expected_stable"],
        expected_prefill_chunk_optimized_hlo_sha256=kwargs["expected_optimized"],
    )
    replay = sealer._replay_batched_graph(stable, optimized, graph="prefill_chunk", args=args)
    assert replay == serialized == report
    assert report["passed"] and not report["dispatch_authorized"]
    assert not report["numerical_claim"] and not report["performance_claim"]


@pytest.mark.parametrize("field,value", [
    ("block_rows", 114), ("block_rows", 128.0), ("graph", "decode"),
    ("prefill_mode", worker.SERIAL_PREFILL_MODE), ("long_context_label", None),
    ("long_context_label", "8k"), ("batched_profile", ""),
    ("expected_stable", "0" * 64),
])
def test_worker_refusal_preserves_originals(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(delivery, "parse_hlo_module", lambda _: pytest.fail("parsed refused evidence"))
    kwargs = values(tmp_path)
    kwargs[field] = value
    with pytest.raises(ValueError):
        worker._write_graph(**kwargs)
    assert (tmp_path / f"{kwargs['graph']}.stablehlo.mlir").read_text() == "untrusted raw"
    assert (tmp_path / f"{kwargs['graph']}.optimized_hlo.txt").read_text() == "untrusted optimized"


def test_long_profile_cannot_enter_short_numerical_or_companion_path():
    with pytest.raises(ValueError):
        admission.short_plan(delivery.PROFILE)
    with pytest.raises(ValueError, match="companion"):
        sealer._replay_batched_graph("", "", graph="decode", args=SimpleNamespace(
            batched_prefill_profile=delivery.PROFILE, context_label="256k_e0",
        ))


def test_sealer_does_not_trust_a_worker_pass(monkeypatch):
    monkeypatch.setattr(delivery, "parse_hlo_module", lambda _: pytest.fail("parsed unbound evidence"))
    with pytest.raises(ValueError, match="StableHLO pin"):
        sealer._replay_batched_graph("", "", graph="prefill_chunk", args=SimpleNamespace(
            batched_prefill_profile=delivery.PROFILE, context_label="256k_e0",
            expected_prefill_chunk_stablehlo_sha256="0" * 64,
            expected_prefill_chunk_optimized_hlo_sha256="0" * 64,
        ))
