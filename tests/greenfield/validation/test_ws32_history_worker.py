"""CPU32 two-branch driver over the frontier core and observer; no TPU or StrategyND.

One subprocess compiles the four frontier programs and the observer once, then
drives three scenarios through a minimal budgeted-call stand-in that honours the
BudgetedCalls contract (dispatch, block_until_ready, preserve before return).
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def test_history_driver_compares_branches_observes_and_requires_reproduction(tmp_path):
    source = r'''
import json
from hashlib import sha256
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield import ws32_history_worker as worker
from scripts.greenfield.ws32_history_frontier import build_program as build_frontier
from scripts.greenfield.ws32_history_observer import build_program as build_observer, observer_config
from glm_tpu.greenfield.runtime import ws32_decoder as dec
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu'
ROOT = Path(ROOT_DIR); SCRATCH = Path(SCRATCH_DIR)
mesh = Mesh(np.asarray(jax.devices(), object)[::-1].reshape(8, 4), ('expert', 'feature'))
raw, weights, wk = fixture(mesh, panel_geometry=True)
raw = replace(raw, context_capacity=protocol.CAPACITY)
production = replace(raw, exact_dsa=True)
observer_cfg = observer_config(raw, strategy_nd_dense=False)
def put(v, s=P()): return jax.device_put(v, NamedSharding(mesh, s))
wk = tuple(put(v) for v in wk)
rope = put(jnp.asarray(dec.build_ws32_main_rope_table(production), jnp.bfloat16))
materializer = dec.build_ws32_exact_dsa_materializer_program(mesh, production)
exact = materializer.promote(materializer.decode(dec.select_ws32_exact_dsa_raw_weights(weights, production)))
layers = weights.layers[:7]

# Both branches share the SAME compiled live-grouping programs, so an identical
# prompt must produce identical histories; the candidate flag is not under test.
frontier = {rows: build_frontier(mesh, raw, block_rows=rows, canonical_dense=False, interpret=True) for rows in (128, 114)}
programs = {'candidate_b128': frontier[128], 'candidate_b114': frontier[114],
            'control_b128': frontier[128], 'control_b114': frontier[114],
            'observer': build_observer(mesh, observer_cfg, sparse_attention_interpret=True, linear_interpret=True)}

class Calls:
    """Minimal BudgetedCalls stand-in: same phase/call/preserve contract, no TPU census."""
    def __init__(self, root):
        self.root = root; root.mkdir(parents=True, exist_ok=True)
        self.programs = dict(programs)
        self.record = {'call_evidence': [], 'jax_process_index': 0}
        self.local_slots = {int(d.id): i for i, d in enumerate(np.asarray(mesh.devices).reshape(-1))}
        self.phases = []
    def phase(self, name, action):
        self.phases.append(name); self.record['current_phase'] = name
        return action()
    def call(self, phase, name, values, *, preserve):
        entry = dict(phase=phase, graph=name, completed=False); self.record['call_evidence'].append(entry)
        result = self.programs[name](*values); jax.block_until_ready(result)
        entry['completed'] = True; preserve(result); return result

prompt = (np.arange(147, dtype=np.int32) * 7 + 30) % 256
plan = protocol.plan(147)
zeros = {b: dict(positions=np.full((4, 1, 128), -1, np.int32), counts=np.zeros((4, 1), np.int32),
                 scores=np.full((4, 1, 128), -np.inf, np.float32)) for b in protocol.BRANCHES}
def run(root, prompt_tokens, originals, **overrides):
    calls = Calls(root)
    kwargs = dict(mesh=mesh, config=production, plan=plan, prompt_tokens=prompt_tokens, embedding=weights.embedding_local,
                  layers=layers, observer_layers=layers, wk=wk, exact=exact, rope=rope, originals=originals,
                  expected_prompt_sha256=sha256(prompt_tokens.tobytes()).hexdigest())
    kwargs.update(overrides)
    error = None
    try: worker.execute_history(calls, **kwargs)
    except ValueError as exc: error = exc
    report = json.loads((root / 'history_report.json').read_text())
    return calls, report, error

# 1. Identical branches with placeholder originals: every boundary/cache equal,
#    observations equal across branches, but reproduction against the
#    placeholder fails and the driver refuses AFTER persisting everything.
calls, report, error = run(SCRATCH / 'run_identical', prompt, zeros)
assert error is not None and 'attribution ineligible' in str(error), error
assert report['complete'] is True and report['attribution_eligible'] is False
assert [c['graph'] for c in calls.record['call_evidence']] == [s.program for s in plan] + ['observer', 'observer']
assert all(c['completed'] for c in calls.record['call_evidence'])
assert len(report['groups']) == 2 and all(g['equal'] for g in report['groups'])
assert report['first_difference'] is None
assert all(all(v.values()) for v in report['cache_equality'].values())
assert report['observations_equal'] is True and report['observer_comparison']['equal'] is True
assert all(report['reproduction'][b]['reproduced'] is False for b in protocol.BRANCHES)
assert set(report['retained']) == {'observer_candidate', 'observer_control'}
assert all(report['observer'][b]['healthy'] for b in protocol.BRANCHES)
with np.load(SCRATCH / 'run_identical/observer_control.npz', allow_pickle=False) as arrays:
    originals = {b: {k: np.asarray(arrays[k]) for k in worker.OBSERVER_FIELDS} for b in protocol.BRANCHES}
assert originals['control']['positions'].shape == (4, 1, 128)
assert worker.reproduce(originals['control'], originals['candidate'])['reproduced']
print('HISTORY_IDENTICAL', flush=True)

# 2. Both branches replay the SAME prompt by design, so a difference must be
#    injected into one branch's computation: only the candidate's first block
#    carries128 live rows, and its token40 is altered there. Expect the first
#    difference at layer0 position40 in group0, exact operands retained once,
#    layer0 KV differing, the control still reproducing its retained original
#    and the candidate failing to.
original_values = worker.block_values
def perturbing(mesh_, tokens, offset, rows, caches, embedding, layers_, wk_, rope_, table, healthy):
    if offset == 0 and rows == 128 and tokens.size == 128:
        tokens = tokens.copy(); tokens[40] = (tokens[40] + 1) % 256
    return original_values(mesh_, tokens, offset, rows, caches, embedding, layers_, wk_, rope_, table, healthy)
with patch.object(worker, 'block_values', perturbing):
    calls, report, error = run(SCRATCH / 'run_perturbed', prompt, originals)
assert error is not None and 'attribution ineligible' in str(error)
assert report['complete'] is True
first = report['first_difference']
assert first['group'] == 0 and first['layer'] == 0 and first['position'] == 40, first
group0 = report['groups'][0]
assert group0['equal'] is False and group0['layers']['0']['update']['first_position'] == 40
assert group0['layers']['0']['update']['differing_rows'] >= 1
assert all(group0['layers']['0'][f]['differing_rows'] == 0 for f in ('route_ids', 'route_weights'))  # dense layer
assert 'first_difference_group0' in report['retained'] and len(report['retained']) == 3
assert report['reproduction']['control']['reproduced'] is True
assert report['reproduction']['candidate']['reproduced'] is False
assert report['cache_equality']['kv']['0'] is False   # row40's own KV changed with its token
with np.load(SCRATCH / 'run_perturbed/first_difference_group0.npz', allow_pickle=False) as arrays:
    a, b = arrays['candidate_layer0_update'], arrays['control_layer0_update']
    assert a.shape == (128, 1024) and (a[:40] == b[:40]).all() and (a[40] != b[40]).any()
print('HISTORY_PERTURBED', flush=True)

# 3. An unhealthy incoming block refuses immediately and files its rows as unhealthy.
original_values = worker.block_values
def poisoned(mesh_, tokens, offset, rows, caches, embedding, layers_, wk_, rope_, table, healthy):
    values = original_values(mesh_, tokens, offset, rows, caches, embedding, layers_, wk_, rope_, table, healthy)
    if offset == 0 and rows == 128 and not getattr(poisoned, 'done', False):
        poisoned.done = True
        return values[:-1] + (put(jnp.bool_(False)),)
    return values
with patch.object(worker, 'block_values', poisoned):
    calls, report, error = run(SCRATCH / 'run_unhealthy', prompt, originals)
assert error is not None and 'unhealthy' in str(error), error
assert report['complete'] is False and report['unhealthy']['step']['index'] == 0
assert set(report['retained']) == {'unhealthy_step0'} and len(calls.record['call_evidence']) == 1
print('HISTORY_UNHEALTHY', flush=True)

# 4. Preflight refusals: wrong plan, foreign prompt digest, incomplete originals.
for bad in (dict(plan=protocol.plan(146)), dict(expected_prompt_sha256='0' * 64),
            dict(originals={'candidate': zeros['candidate']})):
    calls = Calls(SCRATCH / 'run_refused')
    kwargs = dict(mesh=mesh, config=production, plan=plan, prompt_tokens=prompt,
                  embedding=weights.embedding_local, layers=layers, observer_layers=layers, wk=wk,
                  exact=exact, rope=rope, originals=originals,
                  expected_prompt_sha256=sha256(prompt.tobytes()).hexdigest())
    kwargs.update(bad)
    try: worker.execute_history(calls, **kwargs)
    except ValueError: pass
    else: raise AssertionError(f'preflight accepted {sorted(bad)}')
    assert calls.record['call_evidence'] == []
print('HISTORY_WORKER_CPU_PASS', flush=True)
'''
    flags = f"{os.environ.get('XLA_FLAGS', '').strip()} --xla_force_host_platform_device_count=32".strip()
    header = f"ROOT_DIR={str(ROOT)!r}\nSCRATCH_DIR={str(tmp_path)!r}\n"
    result = subprocess.run([sys.executable, "-c", header + source], text=True,
        cwd=ROOT, capture_output=True, timeout=2400,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags, PYTHONPATH=str(ROOT)))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-6000:]
    for marker in ("HISTORY_IDENTICAL", "HISTORY_PERTURBED", "HISTORY_UNHEALTHY", "HISTORY_WORKER_CPU_PASS"):
        assert marker in result.stdout
