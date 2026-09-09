"""Frozen8K host/profile integration without TPU or inherited numerical proof."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_batched_prefill_runner as adapter
from scripts.greenfield import seal_short_decoder_ws32 as sealer

ROOT = Path(__file__).resolve().parents[3]
PROFILE = a.FROZEN_8K_PROFILE


def test_profile_reuses_frozen_graphs_not_numerical_witnesses():
    a.require_acquired_model_source(ROOT, profile=PROFILE)
    assert a.short_acquisition(ROOT, profile=PROFILE) == a.short_acquisition(
        ROOT, profile=a.ROLLED_SHORT_PROFILE
    )
    assert a.short_program_options(PROFILE) == a.short_program_options(
        a.ROLLED_SHORT_PROFILE
    )
    assert a.short_plan(PROFILE).graph_rows == a.ROLLED_PLAN.graph_rows
    assert a.short_plan(PROFILE).split == (63, 91)
    identity = a.short_numerical_identity(profile=PROFILE)
    assert identity["frozen_completion_baseline"]["numerical_inheritance"] is False
    assert identity["frozen_completion_baseline"]["code_hash"] == a.FROZEN_SOURCE_PIN
    assert identity["prefill_budget_seconds"] == 1200
    assert a.short_budget(a.ROLLED_SHORT_PROFILE) == 300
    assert "frozen_completion_baseline" not in a.short_numerical_identity(
        profile=a.ROLLED_SHORT_PROFILE
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("SHORT_DECODER_CONTEXT", "2k"),
        ("PREFILL_CHUNK", "91"),
        ("CONTEXT_CAPACITY", "131072"),
        ("DSA_ADJUDICATION", "1"),
        ("LATER_EVENT_ALARM_ACK", "1"),
        ("EXACT_DSA", "0"),
        ("DECODE_STABLEHLO_SHA", "0" * 64),
    ],
)
def test_profile_refuses_wrong_context_or_inherited_adjudication(field, value):
    env = launch.numerical_environment(profile=PROFILE)
    assert env["GLM_GREENFIELD_WS32_SHORT_DECODER_CONTEXT"] == "8k"
    env["GLM_GREENFIELD_WS32_" + field] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


def test_real64call_adapter_roundtrips_through_sealer(monkeypatch):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_batched_prefill_runner.py")
    )
    args, calls, progress = helpers["fake_workload"](
        monkeypatch, prompt=8155, mlp_window=True, tail_graph_rows=114
    )
    _, token, execution = adapter.execute_graph_pair(
        *args,
        budget_seconds=a.short_budget(PROFILE),
        progress=progress.append,
        fleet_all=bool
    )
    record = dict(
        **a.short_numerical_identity(profile=PROFILE),
        prefill_chunk_length=128,
        context_capacity=8192,
        prefill_execution=execution,
        observed_generated_token_ids=[int(token[0])]
    )
    options = dict(mode="numerical", prompt_length=8155, expected_chunk=128)
    sealer._require_batched_execution(json.loads(json.dumps(record)), **options)
    assert len(calls) == 64 and calls[-1] == ("prefill_tail", 8064, 8155)
    for field, value in (("final_frontier", 8178), ("budget_seconds", 300)):
        bad = deepcopy(record)
        bad["prefill_execution"][field] = value
        with pytest.raises(ValueError):
            sealer._require_batched_execution(bad, **options)


def test_actual_shell_prefix_selects8k_and_tag():
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    prefix = source.split("readonly DSA_ASSOCIATION_URI=", 1)[0]
    env = {
        "PATH": "/usr/bin:/bin",
        "JAX_PLATFORMS": "cpu",
        **launch.numerical_environment(profile=PROFILE),
    }
    result = subprocess.run(
        ["bash", "-c", prefix], env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    tag = "greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_20260909T100000000000000Z"
    sealer._validate_run_tag(
        tag,
        context_label="8k",
        mode="numerical",
        prefill_chunk=128,
        host_main_rope_table=True,
        prefill_mode=adapter.PREFILL_MODE,
        batched_prefill_profile=PROFILE,
    )


def test_input_and_both_builder_static_shapes_remain_frozen(monkeypatch):
    plan = a.short_plan(PROFILE)
    calls = []
    monkeypatch.setattr(
        adapter,
        "build_ws32_batched_prefill_program",
        lambda *args, **kw: calls.append(kw) or object(),
    )
    from types import SimpleNamespace

    adapter.build_graph_pair(
        None,
        SimpleNamespace(context_capacity=8192),
        plan,
        **a.short_program_options(PROFILE)
    )
    assert [x.pop("block_rows") for x in calls] == [128, 114]
    assert calls == [dict(mlp_window=True, **a.short_program_options(PROFILE))] * 2
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: np.asarray(v).copy())
    inputs = adapter.graph_inputs(
        None,
        np.arange(91, dtype=np.int32),
        None,
        None,
        (),
        None,
        mlp_window=True,
        physical_rows=plan.tail_graph_rows,
    )
    assert inputs[0].shape == (114,) and int(inputs[1]) == 91
