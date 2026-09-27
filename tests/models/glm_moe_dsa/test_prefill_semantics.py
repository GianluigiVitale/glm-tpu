"""The production prefill blocks: handoff to decode, atomic refusal, the layer window and the tail.

Semantics of ``glm_tpu.models.glm_moe_dsa.prefill`` on the CPU mesh (32 forced devices, the frozen fixture v1,
Pallas in interpret mode), through the program set the runtime compiles (``build_program_set``: the B128 and
B114 prefill blocks, the packed decode step, the cache initializer). One block embeds its rows once and
visits every layer once (``prefill_layer_window``: four rolled 32-row prefixes and one MLP suffix); it commits
its proposed caches and frontier only when every owner is healthy.

A 32-row block of the same prompt through the same program (32 live rows in a B128 or B114 block) is the
"tiled decoder": a window over 128 rows equals four such blocks, every state leaf bitwise, so the layer-major
window keeps the token-block causal order. Populated prefixes are synthetic device states (random caches, a
permuted page table), not a checkpoint-resume claim.

One child runs every case (the programs compile once) and reports each case's outcome; each test asserts its
own case, so a failing case does not hide the others. Ported from the research package's prefill tests
(``archive/research-20260922``, ``runtime/test_ws32_{batched_prefill,prefill_window,rolled_prefill_tail,
frozen_tail_padding}.py``) onto the production programs; their B17/B32/B33 reference programs and the research
window options no longer exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CHILD = r"""
import functools, json, time, traceback
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.models.glm_moe_dsa import prefill
from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, batched_prefill_state_specs, finish_batched_prefill
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from glm_tpu.runner.hlo_utils import parse_hlo_module
from glm_tpu.runner.programs import build_program_set
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs

mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config, weights, wk, rope = inputs.config, inputs.weights, inputs.wk, inputs.rope
programs = build_program_set(mesh, config, interpret=True)
PREFILL = {rows: spec.fn for rows, spec in programs.prefill.items()}
OTHER = {128: 114, 114: 128}
CAPACITY = config.context_capacity
INT_MIN, INT_MAX = -(2**31), 2**31 - 1
SPECS = batched_prefill_state_specs()
IDS = [(i * 37 + 11) % 256 for i in range(800)]


def put(value, spec=P()):
    return jax.device_put(value, NamedSharding(mesh, spec))


def tokens(ids, rows, pad=-1):
    return put(np.asarray(list(ids) + [pad] * (rows - len(ids)), np.int32))


def run(rows, ids, state, *, count=None, block=None, table=rope, program=None):
    block = tokens(ids, rows) if block is None else block
    count = len(ids) if count is None else count
    fn = PREFILL[rows] if program is None else program
    return jax.block_until_ready(fn(block, put(np.int32(count)), state, weights, wk, table))


def chain(ids, state, pieces, *, table=rope):
    # the tiled decoder: consecutive blocks of the same program, each with ``count`` live rows
    result, start = None, 0
    for rows, count in pieces:
        result = run(rows, ids[start : start + count], state, table=table)
        state, start = result.state, start + count
    assert start == len(ids)
    return result


def same(actual, expected, what=''):
    a, e = jax.tree.leaves(actual), jax.tree.leaves(expected)
    assert len(a) == len(e), what
    for i, (x, y) in enumerate(zip(a, e)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (what, i)


def healthy(result):
    return bool(np.asarray(result.state.decoder.contract_valid).all())


def refused(result, old, what=''):
    assert not np.asarray(result.state.decoder.contract_valid).any(), what
    assert np.asarray(result.next_token).tolist() == [-1], what
    same(result.state, old._replace(decoder=old.decoder._replace(contract_valid=put(np.zeros(1, bool)))), what)


def fresh(prompt_length, position=0):
    state = programs.cache_init.fn(put(np.int32(prompt_length)))
    return state._replace(decoder=state.decoder._replace(
        position=put(np.asarray([position], np.int32)), context_lengths=put(np.asarray([position + 1], np.int32))))


def populated(prompt_length, position, seed):
    # a synthetic populated prefix: random KV and index caches, a permuted page table, repaired keys all 7
    rng = np.random.default_rng(seed)
    state = fresh(prompt_length, position)
    d, s = state.decoder, SPECS.decoder
    decoder = d._replace(
        kv_cache_local=put(np.asarray(jnp.asarray(rng.normal(0, 0.1, d.kv_cache_local.shape), jnp.bfloat16)), s.kv_cache_local),
        index_cache_local=put(np.asarray(jnp.asarray(rng.normal(0, 0.1, d.index_cache_local.shape), jnp.bfloat16)), s.index_cache_local),
        block_tables=put(np.asarray([[2, 0, 1]], np.int32)),
    )
    return state._replace(decoder=decoder, repaired_index_local=jnp.full_like(state.repaired_index_local, 7))


def at(state, **fields):
    return state._replace(decoder=state.decoder._replace(**{k: put(np.asarray(v, np.int32)) for k, v in fields.items()}))


# ---------------------------------------------------------------------------------------------- cases
BASE = populated(705, 505, 517)  # 505 populated rows, then B128 (128 live) and B114 (72 live)


@functools.cache
def two_blocks():
    first = run(128, IDS[:128], BASE)
    second = run(114, IDS[128:200], first.state)
    return first, second


def two_blocks_hand_off_to_decode():
    first, second = two_blocks()
    s = first.state
    assert healthy(first) and np.asarray(first.next_token).tolist() == [-1] and not bool(s.finished)
    assert s.decoder.position.tolist() == [633] and s.decoder.context_lengths.tolist() == [634]
    assert not np.array_equal(np.asarray(s.decoder.index_cache_local), np.asarray(s.repaired_index_local))
    try:
        finish_batched_prefill(first)
        raise AssertionError('decode accepted an unfinished prefill')
    except ValueError:
        pass
    s = second.state
    assert healthy(second) and bool(s.finished) and int(np.asarray(second.next_token)[0]) >= 0
    assert s.decoder.position.tolist() == [705] and s.decoder.context_lengths.tolist() == [706]
    decoder, token = finish_batched_prefill(second)
    np.testing.assert_array_equal(np.asarray(decoder.index_cache_local), np.asarray(s.repaired_index_local))
    step = jax.block_until_ready(programs.decode.fn(token, decoder, weights, rope)).decoded
    assert step.state.position.tolist() == [706] and bool(np.asarray(step.state.contract_valid).all())
    assert 0 <= int(np.asarray(step.next_token)[0]) < 256


def refusals_are_atomic():
    # (a finished state's refusal is test_prefill.py's)
    for count in (0, INT_MAX):
        refused(run(128, IDS[:128], BASE, count=count), BASE, f'count {count}')
    for what, state in (
        ('position', at(BASE, position=[INT_MAX])),
        ('context', at(BASE, context_lengths=[505])),
        ('pages', at(BASE, block_tables=[[2, 2, 1]])),
    ):
        refused(run(128, IDS[:128], state), state, what)
    block = tokens(IDS[:128], 128).at[127].set(256)
    refused(run(128, IDS[:128], BASE, block=block), BASE, 'token 256 in the last row')


def padding_and_repaired_history_do_not_leak():
    first, second = two_blocks()
    # padded ids INT_MIN and NaN rope rows past the live rows (positions >= 705) change nothing
    poisoned = run(114, IDS[128:200], first.state, block=tokens(IDS[128:200], 114, INT_MIN),
                   table=rope.at[705:].set(jnp.nan))
    same(poisoned, second, 'poisoned padding')
    # repaired keys are separate storage: prompt computations never read them
    altered = first.state._replace(repaired_index_local=jnp.full_like(first.state.repaired_index_local, -9))
    changed = run(114, IDS[128:200], altered)
    for name in ('kv_cache_local', 'selected_positions', 'selected_valid_counts', 'selected_scores'):
        np.testing.assert_array_equal(np.asarray(getattr(changed.state.decoder, name)),
                                      np.asarray(getattr(second.state.decoder, name)), name)
    np.testing.assert_array_equal(np.asarray(changed.next_token), np.asarray(second.next_token))


WINDOW_BASE = populated(761, 505, 347)  # two B128 windows


def window_equals_32_row_blocks():
    quarters = ((128, 32), (114, 32), (128, 32), (114, 32))
    first = run(128, IDS[:128], WINDOW_BASE)
    assert healthy(first) and not bool(first.state.finished) and first.state.decoder.position.tolist() == [633]
    same(first, chain(IDS[:128], WINDOW_BASE, quarters), 'first window')
    second = run(128, IDS[128:256], first.state)
    same(second, chain(IDS[128:256], first.state, quarters), 'second window')
    assert healthy(second) and bool(second.state.finished)
    finish_batched_prefill(second)
    # every collective of the compiled window is in the feature-4 or the expert-8 groups
    compiled = PREFILL[128].lower(tokens(IDS[:128], 128), put(np.int32(128)), WINDOW_BASE, weights, wk, rope).compile()
    collectives = parse_hlo_module(compiled.as_text()).collectives
    feature = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
    expert = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
    assert collectives and all(c.replica_groups in (feature, expert) for c in collectives), {
        c.replica_groups for c in collectives} - {feature, expert}
    # live counts that end one row into a tile near the end of the cache, and one row short of the window
    for n, offset in ((33, 1500), (127, 505)):
        state = WINDOW_BASE._replace(prompt_length=put(np.int32(offset + n)))
        state = at(state, position=[offset], context_lengths=[offset + 1])
        table = rope.at[offset + n:].set(jnp.nan)
        actual = run(128, IDS[:n], state, block=tokens(IDS[:n], 128, INT_MIN), table=table)
        assert healthy(actual) and bool(actual.state.finished), n
        assert actual.state.decoder.position.tolist() == [offset + n], n
        pieces = ((114, 32), (114, n - 32)) if n < 64 else ((114, 96), (114, n - 96))
        same(actual, chain(IDS[:n], state, pieces, table=table), f'tail {n}')


def late_failure_rolls_back_every_write():
    # a bad token in the last tile of a window: nothing any earlier tile proposed survives
    block = tokens(IDS[:128], 128).at[110].set(-1)
    refused(run(128, IDS[:128], WINDOW_BASE, block=block), WINDOW_BASE, 'row 110')
    final_base = WINDOW_BASE._replace(prompt_length=put(np.int32(633)))
    control = run(128, IDS[:128], final_base)
    assert healthy(control) and bool(control.state.finished)
    # one owner's late-row health cleared inside the real composition, after the prefix wrote its caches
    original = prefill.prefill_layer_window

    def one_owner_fails_row_110(*args, **kwargs):
        result = original(*args, **kwargs)
        bad = (lax.axis_index('expert') == 3) & (lax.axis_index('feature') == 2)
        return result._replace(contract_valid=result.contract_valid.at[110].set(result.contract_valid[110] & ~bad))

    prefill.prefill_layer_window = one_owner_fails_row_110
    try:
        poisoned = prefill.build_prefill_program(mesh, config, block_rows=128, sparse_attention_interpret=True,
                                                 linear_interpret=True)
        refused(run(128, IDS[:128], final_base, program=poisoned.execute), final_base, 'layer health')
    finally:
        prefill.prefill_layer_window = original
    # one owner arrives unhealthy at a final block: the head's collectives run and nobody commits
    def body(t, c, s, w, k, r):
        bad = (lax.axis_index('expert') == 3) & (lax.axis_index('feature') == 2)
        s = s._replace(decoder=s.decoder._replace(contract_valid=s.decoder.contract_valid & ~bad))
        return prefill.batched_prefill(t, c, s, w, k, r, config=config, sparse_attention_interpret=True,
                                       linear_interpret=True)

    incoming = jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(), SPECS, bf16_weight_specs(config), tuple(P() for _ in wk), P()),
        out_specs=BatchedPrefillResult(SPECS, P()), check_vma=False))
    first, _ = two_blocks()
    refused(run(114, IDS[128:200], first.state, program=incoming), first.state, 'incoming health')


def b114_tail_near_capacity():
    # 91 live rows in B114 from capacity - 128
    start = CAPACITY - 128
    state = fresh(start + 91, start)
    ids = IDS[:91]
    actual = run(114, ids, state, block=tokens(ids, 114, 0))
    assert healthy(actual) and bool(actual.state.finished)
    assert actual.state.decoder.position.tolist() == [start + 91]
    assert actual.state.decoder.context_lengths.tolist() == [start + 92]
    finish_batched_prefill(actual)
    # the padded ids and the rope rows past the live rows (the empty fourth tile included) change nothing
    same(run(114, ids, state, block=tokens(ids, 114, INT_MIN), table=rope.at[start + 91:].set(jnp.nan)), actual,
         'poisoned tail')
    same(chain(ids, state, ((128, 32), (128, 32), (128, 27))), actual, '32/32/27 control')
    # the last live row is row 90, not row 113: a bad one restores the whole initial state
    refused(run(114, ids, state, block=tokens(ids, 114, 0).at[90].set(-1)), state, 'row 90')
    # physical windows past capacity: only live rows write or rotate
    start = CAPACITY - 96
    current = control = fresh(start + 91, start)
    for offset, count, rows in ((0, 32, 128), (32, 32, 128), (64, 27, 114)):
        ids = IDS[offset : offset + count]
        actual = run(rows, ids, current, block=tokens(ids, rows, 0))
        if offset != 32:  # the first window ends past capacity; the last one is the final block
            poisoned = run(rows, ids, current, block=tokens(ids, rows, INT_MIN),
                           table=rope.at[start + offset + count:].set(jnp.nan))
            same(poisoned, actual, f'poisoned piece {offset}')
        expected = run(OTHER[rows], ids, control)
        same(actual, expected, f'piece {offset}')
        if count == 27:
            refused(run(rows, ids, current, block=tokens(ids, rows, 0).at[26].set(-1)), current, 'row 26')
        current, control = actual.state, expected.state
    assert current.decoder.position.tolist() == [start + 91] and bool(current.finished)
    finish_batched_prefill(actual)
    # 33 live rows from capacity - 36: one row past the first tile
    start = CAPACITY - 36
    state = fresh(start + 33, start)
    actual = run(114, IDS[:33], state)
    assert healthy(actual) and bool(actual.state.finished) and actual.state.decoder.position.tolist() == [start + 33]
    same(actual, chain(IDS[:33], state, ((128, 32), (128, 1))), 'static 33')


# The production B114 block with every layer's boundary recorded: prefill.prefill_layer_window is wrapped while
# the program traces, and each layer's inputs and outputs, the incoming state and the committed state become extra
# outputs, every owner's own values (the 32 owners stacked in mesh order). The block's own result is unchanged.
LAYER = ('kv', 'unrepaired', 'repaired', 'selected', 'counts', 'scores', 'wk',
         'kv_out', 'unrepaired_out', 'repaired_out', 'selected_out', 'counts_out', 'scores_out', 'health')
WK_GIVEN = []  # per layer, as traced: whether the layer received a repair wk


@functools.cache
def recording_program():
    original = prefill.prefill_layer_window

    def body(t, c, s, w, k, r):
        seen = []

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            WK_GIVEN.append(args[14] is not None)
            wk_in = jnp.zeros((1,), jnp.float32) if args[14] is None else args[14]
            seen.append(dict(zip(LAYER, (*args[2:8], wk_in, result.cache_local, result.unrepaired_index_cache,
                                         result.repaired_index_cache, result.selected_positions,
                                         result.selected_valid_counts, result.selected_scores,
                                         result.contract_valid))))
            return result

        prefill.prefill_layer_window = record
        try:
            out = prefill.batched_prefill(t, c, s, w, k, r, config=config, sparse_attention_interpret=True,
                                          linear_interpret=True)
        finally:
            prefill.prefill_layer_window = original
        d, n = s.decoder, out.state.decoder
        local = dict(
            before=dict(kv=d.kv_cache_local, index=d.index_cache_local, repaired=s.repaired_index_local),
            after=dict(kv=n.kv_cache_local, index=n.index_cache_local, repaired=out.state.repaired_index_local,
                       selected=n.selected_positions, counts=n.selected_valid_counts, scores=n.selected_scores),
            layers=seen,
        )
        return out, jax.tree.map(lambda v: v[None], local)

    return jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(), SPECS, bf16_weight_specs(config), tuple(P() for _ in wk), P()),
        out_specs=(BatchedPrefillResult(SPECS, P()), P(('expert', 'feature'))), check_vma=False))


def recorded(ids, state):
    out, local = jax.block_until_ready(recording_program()(tokens(ids, 114), put(np.int32(len(ids))), state, weights,
                                                           wk, rope))
    return out, jax.tree.map(np.asarray, local)


def equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    assert actual.tobytes() == expected.tobytes(), what


def wired(local, final):
    # the composition's wiring, on every owner: which buffers and which IndexShare metadata each layer gets, and
    # what the block commits
    before, after, layers = local['before'], local['after'], local['layers']
    slots, producers = config.full_index_slot_by_layer, config.full_index_slots
    assert len(layers) == len(slots) == 8 and slots == (0, 1, 2, None, None, None, 3, None), slots
    assert WK_GIVEN[:8] == [slot is not None for slot in slots], WK_GIVEN
    for layer, (slot, x) in enumerate(zip(slots, layers)):
        if layer == 0:  # nothing selected yet
            assert (x['selected'] == -1).all() and (x['counts'] == 0).all() and np.isneginf(x['scores']).all()
        else:  # the metadata a layer receives is the previous layer's
            for name in ('selected', 'counts', 'scores'):
                equal(x[name], layers[layer - 1][name + '_out'], (layer, name))
        equal(x['kv'], before['kv'][:, layer], (layer, 'kv in'))  # its own KV layer
        equal(after['kv'][:, layer], x['kv_out'], (layer, 'kv out'))
        if slot is None:  # a shared layer keeps the selection it received
            for name in ('selected', 'counts', 'scores'):
                equal(x[name + '_out'], x[name], (layer, name, 'shared'))
        else:  # a full producer reads its own index slot and repair wk, and selects anew
            equal(x['unrepaired'], before['index'][:, slot], (layer, 'unrepaired in'))
            equal(x['repaired'], before['repaired'][:, slot], (layer, 'repaired in'))
            equal(x['wk'], np.broadcast_to(np.asarray(wk[slot]), x['wk'].shape), (layer, 'wk'))
            assert not np.array_equal(x['selected_out'], x['selected']), (layer, 'selection replaced')
    # the witnesses: shared layers 3, 4, 5 carry producer 2's selection, shared layer 7 producer 6's
    for shared, producer in ((3, 2), (4, 2), (5, 2), (7, 6)):
        equal(layers[shared]['selected_out'], layers[producer]['selected_out'], (shared, producer))
        equal(layers[shared]['scores_out'], layers[producer]['scores_out'], (shared, producer))
    # the block writes back exactly each producer's slot: the repaired keys always, the active index the unrepaired
    # keys (an intermediate block) or the repaired ones (the final block)
    for slot, layer in enumerate(producers):
        equal(after['repaired'][:, slot], layers[layer]['repaired_out'], (slot, 'repaired'))
        active = layers[layer]['repaired_out' if final else 'unrepaired_out']
        equal(after['index'][:, slot], active, (slot, 'index'))
    assert not np.array_equal(layers[1]['unrepaired_out'], layers[1]['repaired_out'])
    # the committed IndexShare metadata is the last layer's at the last live row
    return layers


def shared_layers_carry_their_producers_selection():
    # an intermediate block (114 of the 200 prompt rows after BASE's 505) and the final block of two_blocks()
    out, local = recorded(IDS[:114], BASE)
    same(out, run(114, IDS[:114], BASE), 'recording program, intermediate block')
    assert healthy(out) and not bool(out.state.finished)
    layers = wired(local, final=False)
    for name in ('selected', 'counts', 'scores'):
        equal(local['after'][name][:, 0], layers[-1][name + '_out'][:, 113], ('committed', name))
    first, second = two_blocks()
    out, local = recorded(IDS[128:200], first.state)
    same(out, second, 'recording program, final block')
    layers = wired(local, final=True)
    for name in ('selected', 'counts', 'scores'):
        equal(local['after'][name][:, 0], layers[-1][name + '_out'][:, 71], ('committed', name))


def zero_live_tiles_stay_in_capacity():
    # 33 live rows of a B114 block from capacity - 36: the third and fourth 32-row tiles start past the cache and
    # hold no live row; they run at the last in-capacity offset, so every row of every layer, dead rows included,
    # reports healthy (an offset past capacity would make their write invalid)
    start = CAPACITY - 36
    state = fresh(start + 33, start)
    out, local = recorded(IDS[:33], state)
    same(out, run(114, IDS[:33], state), 'recording program near capacity')
    assert healthy(out) and bool(out.state.finished)
    for layer, x in enumerate(local['layers']):
        assert x['health'].all(), (layer, np.argwhere(~x['health']).tolist()[:8])


def reversed_device_order():
    # the fixture bound on a mesh over the devices in reverse order (the research tail tests' mesh), its own B114
    # program: 33 live rows from capacity - 36 equal the 32 + 1 chain on that mesh and the natural mesh's block
    from jax.sharding import Mesh
    from glm_tpu.models.glm_moe_dsa.state import finish_batched_prefill as finish

    from glm_tpu.layers import fp8

    rmesh = Mesh(np.asarray(jax.devices(), object)[::-1].reshape(8, 4), ('expert', 'feature'))
    assert [d.id for d in rmesh.devices.flat] == list(range(31, -1, -1))
    # the per-table weight decode programs are cached by table shape and spec, not by mesh (a process has one mesh):
    # this second mesh binds with an empty cache, and the natural mesh's programs are restored after
    cached = dict(fp8._DECODERS)
    fp8._DECODERS.clear()
    try:
        rinputs = engine_inputs(rmesh, panel_geometry=True)
    finally:
        fp8._DECODERS.clear()
        fp8._DECODERS.update(cached)
    rprograms = build_program_set(rmesh, config, interpret=True)

    def rput(value):
        return jax.device_put(value, NamedSharding(rmesh, P()))

    def rrun(ids, state, block=None):
        block = rput(np.asarray(list(ids) + [-1] * (114 - len(ids)), np.int32)) if block is None else block
        return jax.block_until_ready(rprograms.prefill[114].fn(
            block, rput(np.int32(len(ids))), state, rinputs.weights, rinputs.wk, rinputs.rope))

    start = CAPACITY - 36
    state = rprograms.cache_init.fn(rput(np.int32(start + 33)))
    state = state._replace(decoder=state.decoder._replace(
        position=rput(np.asarray([start], np.int32)), context_lengths=rput(np.asarray([start + 1], np.int32))))
    actual = rrun(IDS[:33], state)
    assert healthy(actual) and bool(actual.state.finished) and actual.state.decoder.position.tolist() == [start + 33]
    finish(actual)
    first = rrun(IDS[:32], state)
    same(actual, rrun(IDS[32:33], first.state), 'reversed: 32 + 1')
    same(actual, run(114, IDS[:33], fresh(start + 33, start)), 'reversed = natural')
    bad = rput(np.asarray(IDS[:33] + [-1] * 81, np.int32)).at[32].set(-1)
    refused(rrun(IDS[:33], state, block=bad), state, 'reversed: row 32')


CASES = (
    two_blocks_hand_off_to_decode,
    refusals_are_atomic,
    padding_and_repaired_history_do_not_leak,
    window_equals_32_row_blocks,
    late_failure_rolls_back_every_write,
    b114_tail_near_capacity,
    shared_layers_carry_their_producers_selection,
    zero_live_tiles_stay_in_capacity,
    reversed_device_order,
)
outcome, seconds = {}, {}
for case in CASES:
    started = time.perf_counter()
    try:
        case()
        outcome[case.__name__] = 'pass'
    except Exception:
        outcome[case.__name__] = traceback.format_exc()
    seconds[case.__name__] = round(time.perf_counter() - started, 1)
print(json.dumps(dict(outcome=outcome, seconds=seconds)))
"""  # noqa: E501 (child program text)


@pytest.fixture(scope="module")
def outcome() -> dict[str, str]:
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])["outcome"]


def check(outcome: dict[str, str], case: str) -> None:
    assert outcome[case] == "pass", outcome[case]


@pytest.mark.cpu32
def test_two_blocks_hand_off_to_decode_cpu32(outcome):
    """B128 then B114 from a populated prefix: the first block leaves repaired keys aside and cannot be decoded;
    the final block promotes them, returns the first token and hands its state to the packed decode step."""
    check(outcome, "two_blocks_hand_off_to_decode")


@pytest.mark.cpu32
def test_refusals_leave_the_state_unchanged_cpu32(outcome):
    """A bad live count, position, context length or page table, or an out-of-vocabulary live token id: health
    false, token -1, every other leaf of the state as it was (a finished state's refusal: ``test_prefill.py``)."""
    check(outcome, "refusals_are_atomic")


@pytest.mark.cpu32
def test_padding_and_repaired_history_do_not_leak_cpu32(outcome):
    """Poisoned padded ids and rope rows change nothing; prompt computations never read the repaired keys."""
    check(outcome, "padding_and_repaired_history_do_not_leak")


@pytest.mark.cpu32
def test_window_equals_four_32_row_blocks_cpu32(outcome):
    """A B128 window equals four 32-live-row blocks (B128 and B114) bitwise, for two windows; tails that end
    inside a tile equal the same rows in two blocks; the compiled window's collectives stay in the feature-4 or
    expert-8 groups."""
    check(outcome, "window_equals_32_row_blocks")


@pytest.mark.cpu32
def test_late_failure_rolls_back_every_write_cpu32(outcome):
    """A bad token in the last tile, one owner's late-row layer health, or one owner arriving unhealthy at a final
    block: every owner refuses the whole block."""
    check(outcome, "late_failure_rolls_back_every_write")


@pytest.mark.cpu32
def test_b114_tail_near_capacity_cpu32(outcome):
    """91 live rows in B114 at the end of the cache (poisoned padding, a 32/32/27 control, row-90 rollback), windows
    that extend past capacity, and 33 live rows one row past the first tile."""
    check(outcome, "b114_tail_near_capacity")


@pytest.mark.cpu32
def test_shared_layers_carry_their_producers_selection_cpu32(outcome):
    """Every layer boundary of an intermediate and a final B114 block, recorded inside the production composition:
    each layer receives the previous layer's IndexShare metadata and its own KV layer; the full producers (layers 0,
    1, 2, 6) read their own index slot and repair ``wk`` and select anew, the shared layers (3, 4, 5, 7) keep what
    they receive (3-5 producer 2's selection, 7 producer 6's); the block commits exactly the producers' slots and the
    last layer's metadata, and equals the production program bitwise."""
    check(outcome, "shared_layers_carry_their_producers_selection")


@pytest.mark.cpu32
def test_zero_live_tiles_past_capacity_stay_healthy_cpu32(outcome):
    """A block whose last two 32-row tiles start past the cache and hold no live row: those tiles run at an
    in-capacity offset, so every row of every layer reports healthy, dead rows included."""
    check(outcome, "zero_live_tiles_stay_in_capacity")


@pytest.mark.cpu32
def test_reversed_device_order_cpu32(outcome):
    """On a mesh over the devices in reverse order, 33 live rows near the end of the cache equal a 32 + 1 chain and
    the natural mesh's block bitwise; a bad last live row rolls the whole state back."""
    check(outcome, "reversed_device_order")
