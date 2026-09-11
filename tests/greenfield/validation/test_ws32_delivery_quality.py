"""§26 task criterion, authenticated original8K replay, and refusal tests.

No TPU or performance claim. This never rewrites the original failed record.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from glm_tpu.greenfield.validation import ws32_delivery_quality as d
from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from glm_tpu.greenfield.validation.ws32_short_context import load_ws32_short_context_oracle, validate_ws32_cache_probe
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_launch as launch

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER = Path('/home/gianl/gcs-models/models/GLM-5.2-FP8')
RECEIPT = ROOT / 'docs/artifacts/prefill-canonical8k-token-refusal-20260909.json'


@pytest.fixture(scope='module')
def original():
    raw = RECEIPT.read_bytes()
    assert sha256(raw).hexdigest() == 'f659925dec463a1473076d913cfd577a81ba0d584c9feba30acb71f5480df175'
    receipt = json.loads(raw)
    # Use the independently authenticated full oracle, not a fabricated gold.
    root = Path('/home/gianl/gcs-models/oracles/greenfield/glm52')
    oracle = load_ws32_short_context_oracle(
        root / 'short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle',
        root / 'short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle',
        expected_token_manifest_sha256=d.TOKEN_MANIFEST_SHA256,
        expected_dsa_manifest_sha256='f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da',
        expected_token_success_sha256='38c0aeb6c4833a0256d4e50152b645e85d24a4f00ca7b2b1731db2d892c5b3cc',
        expected_dsa_success_sha256='0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9',
    )
    folder = Path('/home/gianl/glm-run') / receipt['tag']
    row = receipt['rank_records'][0]
    contents = {}
    for record in row['records'][:2]:
        path = folder / Path(record['name']).name
        data = path.read_bytes()
        assert len(data) == record['size']
        assert sha256(data).hexdigest() == record['sha256']
        contents[path.suffix] = path
    runner = json.loads(contents['.json'].read_text())
    with np.load(contents['.npz'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return oracle, runner, arrays


def test_original_correct_answer_different_prose_is_task_only(original):
    oracle, runner, arrays = original
    result = d.token_result(runner['observed_generated_token_ids'], oracle, tokenizer_root=TOKENIZER)
    assert result['passkey_matches_gold'] is True
    assert result['gold'] == '881446'
    assert result['legacy_diagnostic']['exact_prefix_match'] is False
    assert result['legacy_diagnostic']['first_mismatch']['index'] == 11
    assert result['model_card_quality_claim'] is False
    assert 'exact_prefix_match' not in result
    assert runner['status'] == 'ORACLE_MISMATCH' and runner['correctness_passed'] is False
    for step in range(14):
        check = d.dsa_result(
            producer_layer_ids=arrays['dsa_producer_layer_ids'],
            selected_positions=arrays['dsa_selected_positions'][step],
            selected_valid_counts=arrays['dsa_selected_valid_counts'][step],
            selected_scores=arrays['dsa_selected_scores'][step],
            decode_position=8155 + step, step=step,
        )
        assert check['passed'] and check['cross_oracle'] is False
        assert 'UNSELECTED_SCORE_ROWS_NOT_RECOMPUTED' in check['scope']


@pytest.mark.parametrize('tokens', [[0] * 20, [13] * 20])
def test_wrong_task_answer_fails(original, tokens):
    assert d.token_result(tokens, original[0], tokenizer_root=TOKENIZER)['passkey_matches_gold'] is False


@pytest.mark.parametrize('tokens', [[0] * 19, [True] * 20, [-1] * 20, [154880] * 20])
def test_invalid_tokens_refuse(original, tokens):
    with pytest.raises(ValueError):
        d.token_result(tokens, original[0], tokenizer_root=TOKENIZER)


def test_tokenizer_bytes_and_missing_root_refuse(original, monkeypatch):
    with pytest.raises(ValueError):
        d.token_result([0] * 20, original[0], tokenizer_root=None)
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'changed' if p == TOKENIZER / 'tokenizer.json' else original_read(p))
    with pytest.raises(ValueError, match='tokenizer differs'):
        d.token_result([0] * 20, original[0], tokenizer_root=TOKENIZER)


@pytest.mark.parametrize('mutation', ['empty', 'producer', 'duplicate', 'nan', 'future', 'tie', 'dtype'])
def test_dsa_structural_corruptions_refuse(original, mutation):
    arrays = original[2]
    inputs = dict(
        producer_layer_ids=arrays['dsa_producer_layer_ids'].copy(),
        selected_positions=arrays['dsa_selected_positions'][0].copy(),
        selected_valid_counts=arrays['dsa_selected_valid_counts'][0].copy(),
        selected_scores=arrays['dsa_selected_scores'][0].copy(),
        decode_position=8155, step=0,
    )
    if mutation == 'empty':
        inputs['selected_valid_counts'][:] = 0
        inputs['selected_positions'][:] = -1
        inputs['selected_scores'][:] = -np.inf
    elif mutation == 'producer':
        inputs['producer_layer_ids'][0] = 3
    elif mutation == 'duplicate':
        inputs['selected_positions'][0, 0, 1] = inputs['selected_positions'][0, 0, 0]
    elif mutation == 'nan':
        inputs['selected_scores'][0, 0, 0] = np.nan
    elif mutation == 'future':
        inputs['selected_positions'][0, 0, 0] = 8156
    elif mutation == 'tie':
        inputs['selected_scores'][0, 0, :2] = 1000
        inputs['selected_positions'][0, 0, :2] = [8001, 8000]
    elif mutation == 'dtype':
        inputs['selected_positions'] = inputs['selected_positions'].astype(np.float32)
    assert not d.dsa_result(**inputs)['passed']


def test_profile_keeps_graphs_options_memory_and_distinct_contract():
    env = launch.numerical_environment(profile=d.PROFILE)
    launch.validate_environment(env)
    assert env['GLM_GREENFIELD_WS32_DSA_ADJUDICATION'] == '0'
    assert a.short_plan(d.PROFILE) == a.FROZEN_8K_PLAN
    assert a.short_budget(d.PROFILE) == 1200
    assert a.short_program_options(d.PROFILE) == a.short_program_options(a.CANONICAL_SHORT_PROFILE)
    assert a.short_acquisition(ROOT, profile=d.PROFILE)['graphs'] == a.short_acquisition(ROOT, profile=a.CANONICAL_SHORT_PROFILE)['graphs']
    assert a.short_numerical_identity(profile=d.PROFILE)['validation_contract'] == d.CONTRACT
    assert 'validation_contract' not in a.short_numerical_identity(profile=a.CANONICAL_8K_PROFILE)
    bad = {**env, 'GLM_GREENFIELD_WS32_DSA_ADJUDICATION': '1'}
    with pytest.raises(ValueError):
        launch.validate_environment(bad)
    # Historical pre-correction source MUST still refuse on the current tree.
    with pytest.raises(ValueError, match='model source differs'):
        a.require_acquired_model_source(ROOT, profile=a.SHORT_PROFILE)
    source = (ROOT / 'scripts/greenfield/run_short_decoder_ws32.sh').read_text()
    result = subprocess.run(['bash', '-c', source.split('readonly DSA_ASSOCIATION_URI=', 1)[0]], env={'PATH': '/usr/bin:/bin', **env}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    tag = 'greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26_20260911T230000000000000Z'
    kwargs = dict(context_label='8k', mode='numerical', prefill_chunk=128, context_capacity=8192, host_main_rope_table=True, prefill_mode='layer_major_raw_v1', batched_prefill_profile=d.PROFILE)
    sealer._validate_run_tag(tag, **kwargs)
    with pytest.raises(SystemExit):
        sealer._validate_run_tag(tag.replace('_s26_', '_'), **kwargs)


def test_fresh_import_order():
    for module in ('ws32_prefill_admission', 'ws32_delivery_quality'):
        result = subprocess.run(['/home/gianl/vllm-env/bin/python', '-c', f'import glm_tpu.greenfield.validation.{module}'], env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation', [None, 'wrong_answer', 'contract', 'dsa_bytes', 'cache_bytes'])
def test_composed_eight_rank_sealer_task_contract(tmp_path, monkeypatch, original, mutation):
    # Keep existing explicit synthetic HLO/trace/memory fixtures; use authentic
    # token/DSA originals and the actual task scorer, schema and DB derivation.
    from tests.greenfield.validation.test_ws32_batched_composed_sealer import build

    args, graphs, memories = build(tmp_path, monkeypatch)
    oracle, runner, original_arrays = original
    plan = a.short_plan(d.PROFILE)
    args.batched_prefill_profile = d.PROFILE
    args.prefill_chunk = 128
    args.prefill_budget_seconds = 1200.0
    args.context_label = '8k'
    args.observer_steps = 14
    args.tokenizer_root = TOKENIZER
    args.tag = 'greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26_20260911T230000000000000Z'
    for graph, pins in a.short_acquisition(ROOT, profile=d.PROFILE)['graphs'].items():
        for form, digest in pins.items():
            setattr(args, f'expected_{graph}_{form}', digest)
    monkeypatch.setattr(sealer, 'load_ws32_short_context_oracle', lambda *a, **k: oracle)
    monkeypatch.setattr(sealer, 'validate_ws32_cache_probe', validate_ws32_cache_probe)
    comparison = d.token_result(runner['observed_generated_token_ids'], oracle, tokenizer_root=TOKENIZER)
    dsa = [d.dsa_result(
        producer_layer_ids=original_arrays['dsa_producer_layer_ids'],
        selected_positions=original_arrays['dsa_selected_positions'][step],
        selected_valid_counts=original_arrays['dsa_selected_valid_counts'][step],
        selected_scores=original_arrays['dsa_selected_scores'][step],
        decode_position=8155 + step, step=step,
    ) for step in range(14)]
    for rank in range(8):
        path = tmp_path / f'fleet/runner.rank{rank}.json'
        record = json.loads(path.read_text())
        record.update(a.short_numerical_identity(profile=d.PROFILE))
        record.update(
            prompt_length=8155, prefill_chunk_length=128,
            token_comparison=comparison,
            observed_generated_token_ids=runner['observed_generated_token_ids'],
            dsa_steps=dsa, state=runner['state'], cache_write_probe=runner['cache_write_probe'],
        )
        record['prefill_execution'].update(
            identity=plan.identity(), budget_seconds=1200.0,
            input_transfer_seconds=[.001] * 64, block_wall_seconds=[.1] * 64,
            final_frontier=8155, first_token_ready=runner['observed_generated_token_ids'][0],
        )
        arrays = {name: value.copy() for name, value in original_arrays.items()}
        if mutation == 'wrong_answer':
            record['observed_generated_token_ids'] = [220] + [0] * 28  # preserve first-token binding; forged task PASS must fail
        elif mutation == 'contract' and rank == 7:
            record.pop('validation_contract')
        elif mutation == 'dsa_bytes':
            arrays['dsa_selected_scores'][0, 0, 0, 0] = np.nan
        elif mutation == 'cache_bytes':
            arrays['cache_contract_valid'][:] = False
        target = path.with_suffix('.npz')
        np.savez(target, **arrays)
        record['numerical_tensors'] = dict(
            filename=target.name, byte_count=target.stat().st_size,
            sha256=sealer._digest_file(target),
            arrays={name: dict(dtype=value.dtype.name, shape=list(value.shape),
                              sha256=sha256(value.tobytes()).hexdigest())
                    for name, value in arrays.items()},
        )
        path.write_text(json.dumps(record))
    if mutation:
        reason = {
            'wrong_answer': 'token comparison recomputation',
            'contract': 'runner schema',
            'dsa_bytes': 'DSA recomputation',
            'cache_bytes': 'cache recomputation',
        }[mutation]
        with pytest.raises((SystemExit, ValueError), match=reason):
            sealer._validate(args)
        assert not args.output.exists()
    else:
        assert sealer._validate(args) == 0
        result = json.loads(args.output.read_text())
        assert result['validation_contract'] == d.CONTRACT
        assert result['verified_generated_token_count'] is None
        assert 'S26_TASK_PASSKEY_EXACT' in result['classification']
        assert 'RAW_TOKENS_EXACT' not in result['classification']
        assert 'DSA_CROSS_ORACLE_EXACT' not in result['classification']
        assert 'MODEL_CARD_QUALITY_NOT_ESTABLISHED' in result['classification']
        assert sealer._run_rows(result)[1] == 's26_batched_8k_task_passkey_state_cache'
        assert sealer._run_environment(result)['validation_contract'] == d.CONTRACT
        assert len(graphs) == 7 and memories[0]['owner_count'] == 32
