"""Mode isolation and first-acquisition refusal before costly work."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import (
    UNREGISTERED,
    inspect_ws32_batched_prefill_hlo,
)
from glm_tpu.greenfield.validation import ws32_evidence
from glm_tpu.greenfield.validation.ws32_prefill import (
    PREFILL_MODE,
    SERIAL_PREFILL_MODE,
    BatchedPrefillPlan,
    prefill_graph_kind,
    require_batched_profile,
    require_fleet_prefill_mode,
)
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer


ROOT = Path(__file__).resolve().parents[3]


def profile(**kwargs):
    values = dict(
        exact_dsa=True,
        host_main_rope_table=True,
        block_rows=17,
        long_context=None,
        adjudication_record=None,
        adjudication_sha256="0" * 64,
    )
    values.update(kwargs)
    return values


@pytest.mark.parametrize(
    "change",
    [
        {"exact_dsa": False},
        {"host_main_rope_table": False},
        {"block_rows": 0},
        {"block_rows": 33},
        {"block_rows": True},
        {"long_context": "e0"},
        {"adjudication_record": "serial.json"},
        {"adjudication_sha256": "a" * 64},
    ],
)
def test_batched_profile_refuses_inherited_or_unadmitted_state(change):
    require_batched_profile(PREFILL_MODE, **profile())
    with pytest.raises(ValueError):
        require_batched_profile(PREFILL_MODE, **profile(**change))


def test_mode_geometry_and_fleet_cannot_alias_serial():
    for graph in ("prefill_chunk", "prefill_tail"):
        assert prefill_graph_kind(graph) == "prefill"
        assert prefill_graph_kind(graph, PREFILL_MODE) == "batched_prefill"
    assert prefill_graph_kind("decode", PREFILL_MODE) == "decode"
    with pytest.raises(ValueError):
        prefill_graph_kind("prefill_unknown", PREFILL_MODE)
    old = [{"evidence_layout": ws32_evidence.EVIDENCE_LAYOUT_V2}] * 8
    assert require_fleet_prefill_mode(old) == SERIAL_PREFILL_MODE
    current = [{**r, "prefill_mode": PREFILL_MODE} for r in old]
    assert require_fleet_prefill_mode(current) == PREFILL_MODE
    assert ws32_evidence._runner_layout(current) == ws32_evidence.EVIDENCE_LAYOUT_V2
    with pytest.raises(SystemExit, match="disagree on prefill mode"):
        ws32_evidence._runner_layout(old[:4] + current[4:])
    for invalid in (None, False, "renamed_serial"):
        with pytest.raises(ValueError, match="unknown"):
            require_fleet_prefill_mode([{"prefill_mode": invalid}])
    plan = BatchedPrefillPlan(8155, 17, 8192)
    assert plan.split == (479, 12)


def test_batched_tag_and_serial_sealer_are_separate():
    tag = "greenfield_ws32_short_decoder_8k_acquire_c17_hrope_bp1_20260908T030000000000000Z"
    args = dict(
        context_label="8k", mode="acquire", prefill_chunk=17, host_main_rope_table=True
    )
    sealer._validate_run_tag(tag, **args, prefill_mode=PREFILL_MODE)
    with pytest.raises(SystemExit):
        sealer._validate_run_tag(tag, **args)
    with pytest.raises(SystemExit, match="registered short numerical"):
        sealer._validate(SimpleNamespace(prefill_mode=PREFILL_MODE, batched_prefill_profile=""))
    with pytest.raises(SystemExit, match="cannot authorize batched"):
        sealer._require_prefill_execution(
            {"prefill_mode": PREFILL_MODE},
            mode="acquire",
            prompt_length=8155,
            rank=0,
        )


def test_numerical_refused_before_runtime_or_checkpoint(monkeypatch):
    from tests.greenfield.validation.test_ws32_prefill_admission import request_args

    args = request_args()
    args.compile_only = 0
    args.batched_prefill_profile = ""
    args.delivery_context_label = None
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="not registered"):
        worker.main()


@pytest.mark.parametrize(
    "mode,context,prefill,extra,reason",
    [
        ("numerical", "8k", PREFILL_MODE, {}, "Unknown bounded numerical profile"),
        (
            "acquire",
            "8k",
            PREFILL_MODE,
            {"GLM_GREENFIELD_WS32_DSA_ADJUDICATION": "1"},
            "requires short acquisition",
        ),
        ("acquire", "8k", "typo", {}, "Unknown WS32 prefill"),
        ("numerical", "128k_d0_95", SERIAL_PREFILL_MODE, {}, "refuses new serial"),
        ("acquire", "256k_e0", PREFILL_MODE, {}, "requires short acquisition"),
    ],
)
def test_wrapper_refuses_before_cloud_or_leases(mode, context, prefill, extra, reason):
    # Minimal environment: no credentials inherited and no cloud invocation can
    # be required for these refusals. Every case exits before external commands.
    env = {
        "PATH": "/usr/bin:/bin",
        "JAX_PLATFORMS": "cpu",
        "GLM_GREENFIELD_WS32_SHORT_DECODER": "1",
        "GLM_GREENFIELD_WS32_SHORT_DECODER_MODE": mode,
        "GLM_GREENFIELD_WS32_SHORT_DECODER_CONTEXT": context,
        "GLM_GREENFIELD_WS32_PREFILL_MODE": prefill,
        "GLM_GREENFIELD_WS32_EXACT_DSA": "1",
        "GLM_GREENFIELD_WS32_HOST_MAIN_ROPE_TABLE": "1",
        **extra,
    }
    result = subprocess.run(
        ["/usr/bin/bash", str(ROOT / "scripts/greenfield/run_short_decoder_ws32.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2 and reason in result.stderr


def tiny_hlo():
    return """HloModule diagnostic, num_partitions=32
ENTRY main {
  %p = f32[17] parameter(0)
  ROOT %r = f32[17] copy(%p), metadata={op_name="greenfield_ws32_batched_prefill/layer_0/value"}
}
"""


def test_matched_hlo_pins_cannot_approve_unregistered_graph(tmp_path):
    stable, hlo = "module @main", tiny_hlo()
    report, got_stable, got_hlo = worker._write_graph(
        graph="prefill_tail",
        lowered=SimpleNamespace(compiler_ir=lambda **_: stable),
        compiled=SimpleNamespace(as_text=lambda: hlo),
        hlo_dir=tmp_path,
        expected_stable=sha256(stable.encode()).hexdigest(),
        expected_optimized=sha256(hlo.encode()).hexdigest(),
        hidden_size=6144,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
        prefill_mode=PREFILL_MODE,
        block_rows=17,
    )
    assert (got_stable, got_hlo) == (stable, hlo)
    assert (tmp_path / "prefill_tail.optimized_hlo.txt").read_text() == hlo
    assert report["kind"] == "batched_prefill" and report["live_layer_ids"] == [0]
    assert UNREGISTERED in report["violations"] and report["passed"] is False
    assert report["profile_registered"] is False
    assert "StableHLO identity drifted" not in report["violations"]
    worker._require_graph_authorized(report, compile_only=True)
    with pytest.raises(RuntimeError, match="before execution"):
        worker._require_graph_authorized(report, compile_only=False)
    assert not sealer._graph_valid(report, mode="acquire")
    assert not sealer._graph_valid(report, mode="numerical")


def test_worker_passes_mode_to_remote_and_sealer_without_serial_donation():
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    assert "--prefill-mode '\"$PREFILL_MODE\"'" in source
    assert '--prefill-mode "$PREFILL_MODE"' in source
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.py").read_text()
    assert source.index("batched.bind_raw_prefill_weights(") < source.index(
        "del all_arrays"
    )
    assert "prefill_jits[graph] = prefill_programs[graph].execute" in source
    assert "batched.completed_repair_weights(" in source


def test_expected_acquisition_refusal_preserves_runner_envelope(tmp_path):
    graph = {"passed": False, "violations": [UNREGISTERED]}
    prevalidation = {
        "prefill_mode": PREFILL_MODE,
        "compiled_memory_analysis": {"prefill_chunk": {"temp_size_in_bytes": 1234}},
        "graphs": {name: graph for name in ws32_evidence.EXACT_DSA_GRAPHS},
    }
    output = tmp_path / "runner.rank0.json"
    with pytest.raises(RuntimeError, match="structural violations"):
        worker._publish_acquisition_result(prevalidation, output=output, exact_dsa=True)
    record = json.loads(output.read_text())
    assert record["status"] == "HLO_REFUSED" and record["performance_claim"] is False
    assert all(record[k] == v for k, v in prevalidation.items())
    assert "structural violations" in record["failure"]


def test_serial_acquisition_publication_stays_unchanged(tmp_path):
    graph = {"passed": False, "violations": sorted(worker._VACANT_HLO_VIOLATIONS)}
    prevalidation = {"graphs": {name: graph for name in ws32_evidence.BASE_GRAPHS}}
    output = tmp_path / "runner.rank0.json"
    record = worker._publish_acquisition_result(
        prevalidation, output=output, exact_dsa=False
    )
    assert record == {
        **prevalidation,
        "performance_claim": False,
        "schema_version": 1,
        "status": "HLO_ACQUIRED",
    }
    assert json.loads(output.read_text()) == record
