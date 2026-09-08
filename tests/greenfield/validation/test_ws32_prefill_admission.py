"""Fixed numerical scope; the registration cannot promote serial/long evidence."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from glm_tpu.greenfield.validation.ws32_prefill import BatchedPrefillPlan
from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import UNREGISTERED

ROOT = Path(__file__).resolve().parents[3]


def inputs():
    return dict(
        profile=admission.SHORT_PROFILE,
        plan=admission.SHORT_PLAN,
        reserve_bytes=admission.SHORT_RESERVE_BYTES,
        budget_seconds=admission.SHORT_BUDGET_SECONDS,
        graph_pins=admission.short_acquisition(ROOT)["graphs"],
        repo=ROOT,
    )


def test_registered_profile_matches_original_receipt_and_model_source():
    admission.require_short_numerical_inputs(**inputs())
    admission.require_acquired_model_source(ROOT)


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", ""),
        ("profile", "serial_teacher_forced_v1"),
        ("plan", BatchedPrefillPlan(8155, 17, 8192)),
        ("plan", BatchedPrefillPlan(2034, 17, 131072)),
        ("plan", BatchedPrefillPlan(2034, 32, 8192)),
        ("reserve_bytes", 0),
        ("reserve_bytes", 1073741823),
        ("reserve_bytes", 1073741824.0),
        ("budget_seconds", 3600),
    ],
)
def test_unregistered_workload_refuses(key, value):
    args = inputs()
    args[key] = value
    with pytest.raises(ValueError, match="not registered"):
        admission.require_short_numerical_inputs(**args)


@pytest.mark.parametrize(
    "graph",
    [
        "prefill_chunk",
        "prefill_tail",
        "observer",
        "decode",
        "cache_probe",
        "exact_materialize",
        "exact_promote",
    ],
)
def test_every_graph_pair_must_match_original(graph):
    args = inputs()
    args["graph_pins"][graph]["optimized_hlo_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="all seven"):
        admission.require_short_numerical_inputs(**args)


def test_receipt_drift_is_not_new_registration(tmp_path):
    path = tmp_path / admission.RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt drifted"):
        admission.short_acquisition(tmp_path)


def test_model_source_diff_failure_refuses(monkeypatch):
    def run(command, **kwargs):
        assert command[3:6] == ["diff", "--quiet", admission.ACQUISITION_PIN]
        assert tuple(command[7:]) == admission.MODEL_SOURCE
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(admission.subprocess, "run", run)
    with pytest.raises(ValueError, match="model source differs"):
        admission.require_acquired_model_source(ROOT)


def test_bounded_report_conjunction_cannot_hide_a_structural_refusal():
    # Conjunction test only, not HLO proof. Original integrated replays supply
    # the actual graph report in test_ws32_batched_kernel_hlo.
    report = dict(
        block_rows=17,
        **admission.short_acquisition(ROOT)["graphs"]["prefill_chunk"],
        passed=False,
        profile_registered=False,
        violations=[UNREGISTERED],
    )
    result = admission.authorize_short_graph(
        report, profile=admission.SHORT_PROFILE, repo=ROOT
    )
    assert result["passed"] and not result["runtime_memory_admitted"]
    assert not result["numerical_claim"] and not result["performance_claim"]
    for change in (
        {"violations": [UNREGISTERED, "failed health"]},
        {"passed": True},
        {"block_rows": 12},
        {"optimized_hlo_sha256": "a" * 64},
    ):
        with pytest.raises(ValueError):
            admission.authorize_short_graph(
                {**report, **change}, profile=admission.SHORT_PROFILE, repo=ROOT
            )


def request_args():
    from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE

    return SimpleNamespace(
        prefill_mode=PREFILL_MODE, exact_dsa=1, strategy_nd_dense=1,
        host_main_rope_table=1, rotary_diagnostic=0, prefill_chunk=17,
        long_context=None, dsa_adjudication_record=None,
        dsa_adjudication_sha256="0"*64, context_capacity=8192,
        batched_prefill_profile=admission.SHORT_PROFILE,
        prefill_memory_reserve_bytes=admission.SHORT_RESERVE_BYTES,
        prefill_budget_seconds=admission.SHORT_BUDGET_SECONDS,
        **{f"expected_{g}_{key}": value for g, pins in admission.short_acquisition(ROOT)["graphs"].items() for key,value in pins.items()},
    )


def test_early_request_uses_actual_prompt_and_precedes_runtime():
    import inspect
    from scripts.greenfield import run_short_decoder_ws32 as worker

    admission.require_short_numerical_request(request_args(), prompt_length=2034, repo=ROOT)
    with pytest.raises(ValueError, match="not registered"):
        admission.require_short_numerical_request(request_args(), prompt_length=8155, repo=ROOT)
    source = inspect.getsource(worker.main)
    assert source.index("require_short_numerical_request(args,") < source.index("_initialize_runtime(args)")
    assert source.index("_initialize_runtime(args)") < source.index("load_ws32_runtime_checkpoint(")


@pytest.mark.parametrize("key,value", [
    ("prefill_mode", "serial_teacher_forced_v1"), ("exact_dsa", 0),
    ("host_main_rope_table", 0), ("strategy_nd_dense", 0), ("rotary_diagnostic", 1),
    ("prefill_chunk", 32), ("context_capacity", 131072), ("long_context", "e0"),
    ("dsa_adjudication_record", "serial.json"), ("prefill_budget_seconds", 3600.0),
    ("prefill_memory_reserve_bytes", 0), ("batched_prefill_profile", ""),
    ("expected_decode_stablehlo_sha256", "0"*64),
])
def test_early_request_refuses_unacquired_configuration(key, value):
    args = request_args()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        admission.require_short_numerical_request(args, prompt_length=2034, repo=ROOT)
