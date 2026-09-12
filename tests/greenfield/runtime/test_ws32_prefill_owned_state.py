"""Ownership candidate: CPU mechanism tests, not long TPU HBM admission."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts.greenfield import ws32_prefill_owned_state as candidate


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("bad", [None, object(), {}, lambda: None])
def test_refuses_non_original_program(bad):
    with pytest.raises(TypeError, match="original batched program"):
        candidate.consume_state(bad)


def test_one_worst_graph_preparation_reuses_abstract_inputs(monkeypatch):
    from types import SimpleNamespace
    pair = SimpleNamespace(
        programs={"prefill_chunk": object(), "prefill_tail": object()},
        inputs={"prefill_chunk": (object(),), "prefill_tail": (object(),)},
        manifest_sha256="manifest", source_inventory_sha256="inventory",
    )
    calls = []
    monkeypatch.setattr(candidate.original, "prepare", lambda *a, **kw:
                        calls.append((a, kw)) or pair)
    wrapped = object()
    def wrap(program):
        assert program is pair.programs["prefill_chunk"]
        return wrapped
    monkeypatch.setattr(candidate, "consume_state", wrap)
    result = candidate.prepare_worst_capacity("mesh", "metadata", repo=ROOT)
    assert calls == [(("mesh", "metadata"), dict(repo=ROOT,
        full_canonical=True, long_context_label="256k_e0"))]
    assert result.programs == {candidate.PROGRAM: wrapped}
    assert result.inputs[candidate.PROGRAM] is pair.inputs["prefill_chunk"]
    assert result.manifest_sha256 == "manifest"
    assert result.source_inventory_sha256 == "inventory"


def test_actual_canonical_program_consumed_state_cpu32():
    source = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from scripts.greenfield.ws32_prefill_owned_state import consume_state
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(x,s=P()):return jax.device_put(x,NamedSharding(mesh,s))
wk=tuple(put(x) for x in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
def snapshot(tree):
    return [(x.shape,str(x.dtype),np.asarray(x).tobytes()) for x in jax.tree.leaves(tree)]
def fresh(prompt=633):
    state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=prompt)
    # Populated synthetic prefix, not checkpoint-resume evidence.
    rng=np.random.default_rng(505)
    d=state.decoder
    return state._replace(decoder=d._replace(
        kv_cache_local=put(jnp.asarray(rng.normal(0,.1,d.kv_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
        index_cache_local=put(jnp.asarray(rng.normal(0,.1,d.index_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
        position=put(jnp.array([505],jnp.int32)),
        context_lengths=put(jnp.array([506],jnp.int32))),
        repaired_index_local=put(jnp.full(state.repaired_index_local.shape,7,jnp.bfloat16),P(None,None,'expert',None)))
opts=dict(mlp_window=True,rolled_prefix=True,expert_panels=True,
    paired_position_sort=True,sorted_local_merge=True,canonical_dense=True,
    key_tile=128,sparse_attention_interpret=True,linear_interpret=True)
tokens=put(jnp.arange(128,dtype=jnp.int32)+30)
count=put(jnp.int32(128))
program=b.build_ws32_batched_prefill_program(mesh,config,block_rows=128,**opts)
owned=consume_state(program)
assert owned.original_program is program
initial=fresh(747)
args=(tokens,count,initial,weights,wk,rope)
ordinary=program.execute.lower(*args).compile()
donated=owned.execute.lower(*args).compile()
assert ordinary.memory_analysis().alias_size_in_bytes==0
assert donated.memory_analysis().alias_size_in_bytes>0
print('CPU_ALIAS_BYTES',donated.memory_analysis().alias_size_in_bytes,flush=True)
# Compile metadata must donate exactly all leaves of state, never another arg.
n=len(jax.tree.leaves(initial))
flags=[i.donated for i in jax.tree.leaves(donated.args_info)]
assert flags==[False,False]+[True]*n+[False]*(len(flags)-2-n),flags
retained=snapshot((tokens,count,weights,wk,rope))
expected=snapshot(ordinary(*args))
out=donated(*args);jax.block_until_ready(out)
assert snapshot(out)==expected
assert initial.decoder.kv_cache_local.is_deleted()
assert initial.decoder.index_cache_local.is_deleted()
assert initial.repaired_index_local.is_deleted()
try:np.asarray(initial.decoder.kv_cache_local);raise AssertionError('old cache usable')
except RuntimeError:pass
assert snapshot((tokens,count,weights,wk,rope))==retained
assert out.state.decoder.position.tolist()==[633] and not bool(out.state.finished)
# B114 tail with only three live rows, as in128K. Input padding/NaN RoPE ignored.
tail_program=b.build_ws32_batched_prefill_program(mesh,config,block_rows=114,**opts)
tail=put(jnp.concatenate((jnp.array([77,78,79],jnp.int32),jnp.full(111,-1,jnp.int32))))
tail_rope=rope.at[636:747].set(jnp.nan)
tail_state=out.state._replace(prompt_length=put(jnp.int32(636)))
tail_args=(tail,put(jnp.int32(3)),tail_state,weights,wk,tail_rope)
expected=snapshot(tail_program.execute(*tail_args))
last=consume_state(tail_program).execute(*tail_args);jax.block_until_ready(last)
assert snapshot(last)==expected
b.finish_ws32_batched_prefill(last)
assert last.state.decoder.position.tolist()==[636]
assert snapshot(last.state.decoder.index_cache_local)==snapshot(last.state.repaired_index_local)
# Refusal returns the old values with false health, even though inputs consumed.
for bad in ('token','count','incoming_health'):
    state=fresh()
    if bad=='incoming_health':
        state=state._replace(decoder=state.decoder._replace(contract_valid=put(jnp.array([False]))))
    expected_state=snapshot(state._replace(decoder=state.decoder._replace(contract_valid=put(jnp.array([False])))))
    t=tokens.at[100].set(-1) if bad=='token' else tokens
    c=put(jnp.int32(129)) if bad=='count' else count
    args=(t,c,state,weights,wk,rope)
    expected=snapshot(ordinary(*args))
    result=donated(*args);jax.block_until_ready(result)
    assert snapshot(result)==expected and snapshot(result.state)==expected_state
    assert not np.asarray(result.state.decoder.contract_valid).any()
    assert result.next_token.tolist()==[-1]
    assert state.decoder.kv_cache_local.is_deleted()
    assert snapshot((tokens,count,weights,wk,rope))==retained
    try:b.finish_ws32_batched_prefill(result);raise AssertionError('served refusal')
    except ValueError:pass
print('OWNED_STATE_CPU32_PASS',flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True,
        timeout=480, env=dict(os.environ, JAX_PLATFORMS="cpu",
                             XLA_FLAGS="--xla_force_host_platform_device_count=32"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
    assert "OWNED_STATE_CPU32_PASS" in result.stdout


@pytest.mark.parametrize("refuse", [False, True])
def test_existing_host_loop_never_reads_consumed_state(monkeypatch, refuse):
    """Real donated CPU arrays and host loop; model/memory/fleet are fixtures.

    This is deliberately NOT a numerical admission profile. Production's old
    memory guard must continue refusing aliases until a distinct profile exists.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        Ws32BatchedPrefillResult, Ws32BatchedPrefillState,
    )
    from scripts.greenfield import ws32_batched_prefill_runner as adapter
    from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config
    assert jax.default_backend() == "cpu"
    plan = adapter.BatchedPrefillPlan(35, 17, 8192)
    held = []
    calls = []
    def fresh(mesh, cfg, *, prompt_length):
        d = Ws32DecoderState(
            jnp.zeros(4), jnp.ones(4), jnp.zeros((1, 1), jnp.int32),
            jnp.zeros(1, jnp.int32), jnp.zeros((1, 1)),
            jnp.array([0], jnp.int32), jnp.array([[0]], jnp.int32),
            jnp.array([1], jnp.int32), jnp.array([True]),
        )
        return Ws32BatchedPrefillState(d, jnp.full(4, 2.),
                                      jnp.int32(prompt_length), jnp.bool_(False))
    def body(tokens, count, state, weights, wk, rope):
        end = state.decoder.position + count
        final = end[0] == state.prompt_length
        d = state.decoder._replace(
            kv_cache_local=state.decoder.kv_cache_local + 1,
            position=end, context_lengths=end + 1,
            contract_valid=jnp.array([not refuse]),
        )
        return Ws32BatchedPrefillResult(state._replace(decoder=d, finished=final),
                                       jnp.where(final, 123, -1)[None])
    fn = jax.jit(body, donate_argnums=(2,))
    def dispatch(*args):
        # Hold even stale Python references to prove the caller does not read
        # them. Old cache handles really become unusable after each dispatch.
        held.append(args[2].decoder.kv_cache_local)
        result = fn(*args)
        jax.block_until_ready(result)
        assert held[-1].is_deleted()
        calls.append(True)
        return result
    monkeypatch.setattr(adapter, "make_ws32_batched_prefill_state", fresh)
    monkeypatch.setattr(adapter, "replicated", lambda mesh, v: jnp.asarray(v))
    monkeypatch.setattr(adapter, "make_prefill_memory_record", lambda *a, **k: {})
    monkeypatch.setattr(adapter, "validate_prefill_memory_record", lambda _: None)
    def run():
        return adapter.execute_graph_pair(
            None, config(), plan, np.arange(35, dtype=np.int32),
            {name: dispatch for name in adapter.GRAPHS}, jnp.ones(1),
            (jnp.ones(1),), jnp.ones(1), budget_seconds=60,
            required_memory_reserve_bytes=1 << 30, progress=lambda _: None,
            fleet_all=bool,
        )
    if refuse:
        with pytest.raises(RuntimeError, match="unhealthy/wrong frontier"):
            run()
        assert len(calls) == 1
    else:
        decoder, token, record = run()
        assert len(calls) == 3 and token.tolist() == [123]
        assert decoder.position.tolist() == [35]
        assert record["ttft_measured"] is False
    assert all(x.is_deleted() for x in held)


def test_production_256k_abstract_owned_signature():
    """Real authenticated metadata/lowering; forbid weights, placement/compile."""
    source = r'''
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax, numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_canonical_prefill_compile as metadata_source
from scripts.greenfield import ws32_prefill_owned_state as candidate
assert jax.default_backend()=='cpu'
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
old=Path.open
def checked(path,*args,**kwargs):
    assert path.suffix not in ('.safetensors','.bin'),path
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json','SUCCESS'),path
    return old(path,*args,**kwargs)
with patch.object(Path,'open',checked), \
     patch('jax.device_put',side_effect=AssertionError('placement')), \
     patch('jax.stages.Lowered.compile',side_effect=AssertionError('compile')):
    metadata=metadata_source.read_metadata(Path.cwd())
    pair=candidate.prepare_worst_capacity(mesh,metadata,repo=Path.cwd())
    assert set(pair.programs)==set(pair.inputs)=={candidate.PROGRAM}
    values=pair.inputs[candidate.PROGRAM]
    ids,count,state,weights,wk,rope=values
    assert ids.shape==(128,) and len(weights.layers)==78
    assert state.decoder.block_tables.shape==(1,513) and rope.shape==(262656,64)
    assert all(isinstance(x,jax.ShapeDtypeStruct) and x.sharding is not None
               for x in jax.tree.leaves(values))
    traced=pair.programs[candidate.PROGRAM].execute.trace(*values)
    assert traced.donate_argnums==tuple(range(2,2+len(jax.tree.leaves(state))))
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        lowered=traced.lower(lowering_platforms=('tpu',))
        module=lowered.compiler_ir('stablehlo')
        raw=str(module).encode()
    print('OWNED_E0_RAW',len(raw),sha256(raw).hexdigest(),flush=True)
    main=next(op for op in module.body.operations
              if op.operation.name=='func.func' and op.attributes['sym_name'].value=='main')
    # The partitioned graph defers assignment to XLA with jax.buffer_donor;
    # tf.aliasing_output is only the already-assigned representation. Neither
    # proves actual TPU reuse; check precise input ownership in either form.
    donors=tuple(i for i,a in enumerate(main.attributes['arg_attrs'])
                 if 'jax.buffer_donor' in a or 'tf.aliasing_output' in a)
    assert donors==traced.donate_argnums,donors
    assert b'262656x64xbf16' in raw
    print('OWNED_E0_ABSTRACT_PASS',flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True,
        timeout=300, env=dict(os.environ, JAX_PLATFORMS="cpu",
                             XLA_FLAGS="--xla_force_host_platform_device_count=32"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
    assert "OWNED_E0_ABSTRACT_PASS" in result.stdout
