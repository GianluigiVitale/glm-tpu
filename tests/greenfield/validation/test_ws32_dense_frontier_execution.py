"""Real compiler writer/journal/BudgetedCalls, fixture device math and counters."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import ws32_dense_frontier_execution as execution
from scripts.greenfield import ws32_dense_frontier_prepare as preparation
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_worker as worker
from tests.greenfield.validation.test_ws32_dense_frontier_worker import setup as worker_setup

ROOT = Path('/home/gianl/glm-run')
ORIGINAL = ROOT / protocol.ORIGINAL_TAG / 'first_window_collected'
SLOTS = {9: 0, 13: 1, 25: 2, 29: 3}


def array(value, *, sharded=False):
    shards = []
    for device, slot in SLOTS.items():
        index = (slice(None), slice(None))
        if sharded:
            width = value.shape[1] // 4
            index = (slice(None), slice(slot % 4 * width, (slot % 4 + 1) * width))
        shards.append(NS(device=NS(id=device, process_index=3, platform='tpu'),
                         data=value[index].copy(), index=index))
    return NS(shape=value.shape, dtype=value.dtype, addressable_shards=shards)


def wk_inputs():
    return (array(np.zeros((128, 6144), np.uint8), sharded=True),
            array(np.ones((1, 48), np.float32), sharded=True))


def test_actual_db604_prompt_rope_and_original_dsa_pins(monkeypatch):
    from glm_tpu.greenfield.types import ModelGeometry
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
    import jax
    repo = Path(__file__).resolve().parents[3]
    receipt = json.loads((repo / 'docs/artifacts/prefill-frozen-own8k-token-refusal-20260909.json').read_text())
    original = ROOT / receipt['tag'] / 'runner.rank0.json'
    raw = original.read_bytes()
    assert sha256(raw).hexdigest() == receipt['hosts'][0]['originals']['json']['sha256'] == execution.DSA_PIN_SOURCE_SHA
    source = json.loads(raw)
    assert execution.DSA_PINS == {f'expected_dsa_{suffix}': source[f'dsa_oracle_{suffix}']
                                  for suffix in ('manifest_sha256', 'success_sha256')}
    prior = json.loads((ORIGINAL / 'host_records/runner.rank0.json').read_text())
    assert 'dsa_oracle_manifest_sha256' not in prior
    manifest = json.loads(Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/manifest.json').read_text())
    config = Ws32DecoderConfig(ModelGeometry.from_dict(manifest['geometry']), 8192,
                               host_main_rope_table=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('host inputs initialized a device')
    monkeypatch.setattr(jax, 'devices', forbidden)
    monkeypatch.setattr(jax, 'device_put', forbidden)
    tokens, rope = execution.host_inputs(prior, config)
    assert tokens.shape == (8155,) and rope.shape == (8192, 64)
    prior['main_rope_table']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='prompt/RoPE'):
        execution.host_inputs(prior, config)


@pytest.mark.parametrize('mutation', [None, 'index', 'process', 'platform', 'missing',
                                      'dtype', 'global_shape', 'arity', 'nonfinite', 'inexact'])
def test_actual_wk_capsules_and_owner_boundaries(tmp_path, mutation):
    inputs = wk_inputs()
    decoded = array(np.zeros((128, 6144), ml_dtypes.bfloat16))
    record = dict(jax_process_index=3)
    if mutation == 'index': inputs[0].addressable_shards[0].index = (slice(None), slice(1536, 3072))
    elif mutation == 'process': inputs[0].addressable_shards[0].device.process_index = 7
    elif mutation == 'platform': decoded.addressable_shards[0].device.platform = 'cpu'
    elif mutation == 'missing': decoded.addressable_shards.pop()
    elif mutation == 'dtype': inputs[0].dtype = np.float32
    elif mutation == 'global_shape': inputs[0].shape = (128, 1536)
    elif mutation == 'arity': inputs = inputs[:1]
    elif mutation == 'nonfinite': decoded.addressable_shards[0].data[0, 0] = np.nan
    if mutation not in (None, 'inexact'):
        with pytest.raises(ValueError):
            execution.capture_wk(tmp_path, record, SLOTS, layer=0, name='wk_decode', value=decoded, inputs=inputs)
        assert (tmp_path / 'layer0_wk_decode.npz').exists() == (mutation == 'nonfinite')
        if mutation == 'nonfinite': assert record['wk_originals']['layer0/wk_decode']['valid'] is False
        return
    execution.capture_wk(tmp_path, record, SLOTS, layer=0, name='wk_decode', value=decoded, inputs=inputs)
    promoted = array(np.zeros((128, 6144), np.float32))
    if mutation == 'inexact': promoted.addressable_shards[0].data[0, 0] = 1
    def promote():
        execution.capture_wk(tmp_path, record, SLOTS, layer=0, name='wk_promote', value=promoted, inputs=(decoded,))
    if mutation:
        with pytest.raises(ValueError, match='originals preserved'): promote()
        assert not record['wk_originals']['layer0/wk_promote']['valid']
    else:
        promote()
        for name, report in record['wk_originals'].items():
            path = tmp_path / (name.replace('/', '_') + '.npz')
            assert sha256(path.read_bytes()).hexdigest() == report['npz_sha256']
            assert report['valid']
        with pytest.raises(ValueError, match='replace'): promote()


def staged(tmp_path, monkeypatch, failure=None):
    initial, config, prompt, events = worker_setup(tmp_path, monkeypatch,
                                                   'reproduction' if failure == 'reproduction' else None)
    record = initial.record
    record.update(protocol=protocol.PROTOCOL, programs={}, diagnostic_only=True,
                  code_hash='a'*40, launch_rank=0)
    dense = initial.programs['dense01']
    inputs = [wk_inputs(), wk_inputs()]
    inputs[1][0].addressable_shards[0].data[0, 0] = 1
    layers = tuple(NS(dsa=NS(wk_bits_local=v[0], wk_scale_local=v[1])) for v in inputs)

    class Program:
        memory_analysis = dense.memory_analysis
        def __init__(self, name): self.name = name
        def as_text(self): return self.name + ' optimized fixture'
        def __call__(self, *args):
            if self.name == 'dense01': return dense(*args)
            events.append(('wk_dispatch', self.name, args[0] is inputs[1][0]))
            if failure == 'wk_dispatch': raise ValueError('fixture WK failure')
            result = array(np.zeros((128, 6144), ml_dtypes.bfloat16 if self.name == 'wk_decode' else np.float32))
            if failure == 'wk_capture': result.addressable_shards[0].data[0, 0] = np.nan
            return result

    class Lowered:
        def __init__(self, name): self.name = name
        def compiler_ir(self, *, dialect):
            assert dialect == 'stablehlo'
            return self.name + ' stable fixture'
        def compile(self):
            events.append(('compile', self.name))
            if failure == 'compile_' + self.name: raise ValueError('fixture compiler failure')
            return Program(self.name)

    jobs = tuple((name, NS(lower=lambda *args, name=name: Lowered(name)), ()) for name in worker.PROGRAMS)
    monkeypatch.setattr(preparation, 'compiler_programs', lambda *args: jobs)
    def inspect(name, stable, optimized, memory):
        events.append(('admit', name))
        assert all((tmp_path / f'{g}.optimized_hlo.txt').is_file() for g in worker.PROGRAMS)
        assert stable == name + ' stable fixture'
        if failure == 'admission': raise ValueError('fixture actual HLO refusal')
        return dict(passed=True, fixture=True)
    def consensus(ok):
        events.append(('vote', ok))
        if failure == 'peer_admission' and events[-2:-1] == [('admit', 'dense01')]: return False
        return ok
    if failure == 'finalize':
        original_close = execution.DenseJournal.close
        def close(self):
            original_close(self)
            raise ValueError('fixture journal finalization failure')
        monkeypatch.setattr(execution.DenseJournal, 'close', close)
    def run():
        execution.execute(root=tmp_path, record=record, mesh=None, prepared=NS(config=config),
                          embedding=None, layers=layers, tokens=prompt, rope=None, witness={},
                          local_slots=SLOTS, consensus=consensus, inspect_program=inspect)
    return run, record, events


def test_actual_writer_journal_nine_budgeted_calls_and_capture(tmp_path, monkeypatch):
    run, record, events = staged(tmp_path, monkeypatch)
    run()
    assert len(record['call_evidence']) == 9 and all(v['completed'] for v in record['call_evidence'])
    assert [e[0] for e in events if e[0] in ('compile', 'admit')][:6] == ['compile']*3 + ['admit']*3
    assert [e[2] for e in events if e[0] == 'wk_dispatch'] == [False, False, True, False]
    journal = [json.loads(line) for line in (tmp_path / 'compile_journal.jsonl').read_text().splitlines()]
    assert [v['graph'] for v in journal if v['stage'] == 'inspected'] == list(worker.PROGRAMS)
    assert record['compile_journal_sha256'] == sha256((tmp_path / 'compile_journal.jsonl').read_bytes()).hexdigest()
    assert len(list(tmp_path.glob('*.npz'))) == 9
    assert record['dense_frontier']['comparison']['reproduced']
    assert not record['dense_frontier']['numerical_promotion']


@pytest.mark.parametrize('failure', ['compile_wk_decode', 'compile_wk_promote', 'compile_dense01',
                                    'admission', 'peer_admission', 'wk_dispatch', 'wk_capture',
                                    'reproduction', 'finalize'])
def test_compiler_and_runtime_failure_preservation(tmp_path, monkeypatch, failure):
    run, record, events = staged(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError)): run()
    assert record['status'] == 'DIAGNOSTIC_FAILED'
    assert record.get('error') or record.get('finalization_error')
    assert record['acquisition_phases']['dense/finalize']['status'] in ('COMPLETE', 'FAILED')
    model_calls = [e for e in events if e[0] == 'dense_dispatch']
    assert len(model_calls) == (5 if failure in ('reproduction', 'finalize') else 0)
    if failure.startswith('compile') or failure in ('admission', 'peer_admission'):
        assert not [e for e in events if e[0] == 'wk_dispatch']
    if failure in ('admission', 'peer_admission'):
        assert all((tmp_path / f'{g}.optimized_hlo.txt').is_file() for g in worker.PROGRAMS)
    if failure == 'wk_capture':
        assert (tmp_path / 'layer0_wk_decode.npz').exists()
        assert not record['wk_originals']['layer0/wk_decode']['valid']
    if failure == 'reproduction': assert (tmp_path / 'narrow_128.npz').exists()
