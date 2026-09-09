"""Same-graph live-window diagnostic: host accounting, never promotion."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import subprocess

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_batched_prefill_runner as adapter
from scripts.greenfield import seal_short_decoder_ws32 as sealer

ROOT = Path(__file__).resolve().parents[3]
PROFILE = a.FROZEN_LIVE32_PROFILE


def test_same_graphs_options_source_but_distinct_diagnostic_identity():
    plan = a.short_plan(PROFILE)
    assert plan.split == (254, 27)
    assert plan.stride_rows == 32 and plan.block_rows == 128
    assert plan.graph_rows == a.FROZEN_8K_PLAN.graph_rows
    assert a.short_acquisition(ROOT, profile=PROFILE) == a.short_acquisition(
        ROOT, profile=a.FROZEN_8K_PROFILE
    )
    assert a.short_program_options(PROFILE) == a.short_program_options(a.FROZEN_8K_PROFILE)
    a.require_acquired_model_source(ROOT, profile=PROFILE)
    identity = a.short_numerical_identity(profile=PROFILE)
    assert identity['live_window_diagnostic']['diagnostic_only'] is True
    assert identity['live_window_diagnostic']['completion_baseline_replacement'] is False
    assert identity['frozen_completion_baseline']['numerical_inheritance'] is False
    assert 'live_block_rows' not in a.FROZEN_8K_PLAN.identity()
    assert a.FROZEN_8K_PLAN.split == (63, 91)
    assert a.short_budget(PROFILE) == 1200


@pytest.mark.parametrize('live', [True, 0, -1, 129, 32.0])
def test_live_stride_strictly_bounded(live):
    with pytest.raises(ValueError, match='live stride'):
        replace(a.FROZEN_LIVE32_PLAN, live_block_rows=live)


def test_unregistered_stride_refuses_before_launch():
    for plan in [a.FROZEN_8K_PLAN, replace(a.FROZEN_LIVE32_PLAN, live_block_rows=16)]:
        with pytest.raises(ValueError, match='not registered'):
            a.require_short_numerical_inputs(
                profile=PROFILE, plan=plan, reserve_bytes=a.SHORT_RESERVE_BYTES,
                budget_seconds=1200,
                graph_pins=a.short_acquisition(ROOT, profile=PROFILE)['graphs'], repo=ROOT,
            )


def test_actual255call_host_adapter_and_sealer_accounting(monkeypatch):
    helpers = runpy.run_path(str(ROOT/'tests/greenfield/runtime/test_ws32_batched_prefill_runner.py'))
    args, calls, progress = helpers['fake_workload'](
        monkeypatch, prompt=8155, mlp_window=True, tail_graph_rows=114, live_block_rows=32,
    )
    _, token, execution = adapter.execute_graph_pair(
        *args, budget_seconds=1200, progress=progress.append, fleet_all=bool,
    )
    assert len(calls) == 255
    assert calls[0] == ('prefill_chunk', 0, 32)
    assert calls[-2] == ('prefill_chunk', 8096, 8128)
    assert calls[-1] == ('prefill_tail', 8128, 8155)
    record = dict(
        **a.short_numerical_identity(profile=PROFILE), prefill_chunk_length=128,
        context_capacity=8192, prefill_execution=execution,
        observed_generated_token_ids=[int(token[0])],
    )
    options = dict(mode='numerical', prompt_length=8155, expected_chunk=128)
    sealer._require_batched_execution(json.loads(json.dumps(record)), **options)
    for mutate in ('profile', 'stride', 'frontier', 'calls'):
        bad = deepcopy(record)
        if mutate == 'profile': bad['batched_prefill_profile'] = a.FROZEN_8K_PROFILE
        elif mutate == 'stride': bad['batched_prefill_plan']['live_block_rows'] = 16
        elif mutate == 'frontier': bad['prefill_execution']['final_frontier'] = 8178
        else: bad['prefill_execution']['block_wall_seconds'].pop()
        with pytest.raises(ValueError): sealer._require_batched_execution(bad, **options)


def test_diagnostic_cannot_seal_even_if_worker_numerically_passes():
    args = SimpleNamespace(prefill_mode=adapter.PREFILL_MODE, batched_prefill_profile=PROFILE)
    with pytest.raises(SystemExit, match='diagnostic cannot seal'):
        sealer._validate(args)


@pytest.mark.parametrize('mutation', [None, 'decode_vacancy', 'wrong_main_stable'])
def test_actual_worker_startup(tmp_path, monkeypatch, mutation):
    helpers = runpy.run_path(str(ROOT/'tests/greenfield/runtime/test_ws32_short_actual_startup.py'))
    helpers['test_actual_startup_recipe_reaches_post_pin_sentinel'](
        tmp_path, monkeypatch, 'worker', PROFILE, mutation,
    )


def test_actual_shell_profile_and_distinct_tag():
    env = {'PATH':'/usr/bin:/bin', 'JAX_PLATFORMS':'cpu', **launch.numerical_environment(profile=PROFILE)}
    source = (ROOT/'scripts/greenfield/run_short_decoder_ws32.sh').read_text()
    result = subprocess.run(['bash','-c',source.split('readonly DSA_ASSOCIATION_URI=',1)[0]],
                            env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stderr
    sealer._validate_run_tag(
        'greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_live32_20260909T120000000000000Z',
        context_label='8k', mode='numerical', prefill_chunk=128,
        host_main_rope_table=True, prefill_mode=adapter.PREFILL_MODE,
        batched_prefill_profile=PROFILE,
    )
