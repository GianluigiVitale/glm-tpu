"""Actual explicit assembly executables on CPU32; no model or TPU claims."""

import os
import subprocess
import sys


def test_explicit_device_assembly_matches_existing_operations_cpu32():
    code = r"""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield import prefill_completed_window as old
from scripts.greenfield import prefill_completed_window_assembly as new
assert jax.default_backend() == 'cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
rep=NamedSharding(mesh,P()); feat=NamedSharding(mesh,P(None,'feature')); hp=NamedSharding(mesh,P('expert','feature',None))
def put(value,spec=rep):return jax.device_put(value,spec)
def scalar(value):return put(np.asarray(value,np.int32))
rng=np.random.RandomState(5)
base=put(np.asarray(rng.randn(128,6144)*.01,dtype='bfloat16'),feat)
rows=(base,base,put(np.tile(np.arange(2048,dtype=np.int32),(128,1))),put(np.full(128,2048,np.int32)),put(np.zeros((128,2048),np.float32)),put(np.ones((8,4,128),bool),hp),put(np.ones((128,64),dtype='bfloat16')))
# No cache or weight is allocated or allowed into assembly executable arguments.
cache=SimpleNamespace(shape=(8,8,64,640));index=object();repair=object();dense=object();moe=object()
values=[None]*20
for i,v in zip(new.ROW_INPUT_INDICES,rows):values[i]=v
values[2:5]=[cache,index,repair]; values[8:10]=[scalar(2553),scalar(33)];values[16:18]=[dense,moe]
values=tuple(values); functions=new.build_programs(mesh)
args=new.prefix_arguments(values,scalar(0))
lowered={"prepare_prefix":functions['prepare_prefix'].lower(*args)}
compiled={"prepare_prefix":lowered['prepare_prefix'].compile()}
prefixes=[]
for tile in range(4):
 prepared=compiled['prepare_prefix'](*new.prefix_arguments(values,scalar(tile)));jax.block_until_ready(prepared)
 actual=new.attach_prefix(values,prepared,prefixes[-1] if prefixes else None)
 expected=old.prefix_inputs(values,tile,prefixes[-1] if prefixes else None)
 for i in (*new.ROW_INPUT_INDICES,8,9):np.testing.assert_array_equal(actual[i],expected[i])
 assert actual[16] is dense and actual[17] is moe
 assert all(actual[i] is expected[i] for i in (2,3,4))
 prefixes.append((actual[0],actual[1],cache,index,repair,actual[5],actual[6],actual[7],actual[18],actual[0]))
prefixes=tuple(prefixes)
wargs=new.suffix_arguments(prefixes,values[9])
lowered['prepare_wide']=functions['prepare_wide'].lower(*wargs)
compiled['prepare_wide']=lowered['prepare_wide'].compile()
prepared=compiled['prepare_wide'](*wargs);jax.block_until_ready(prepared)
actual=new.attach_suffix(values,prepared);expected=old.suffix_inputs(prefixes,values[9],dense,moe)
for i in (0,1,4):np.testing.assert_array_equal(actual[i],expected[i])
assert actual[2] is dense and actual[3] is moe
nargs=new.suffix_arguments((prefixes[0],),values[9],scalar(0))
lowered['prepare_narrow']=functions['prepare_narrow'].lower(*nargs)
compiled['prepare_narrow']=lowered['prepare_narrow'].compile()
suffixes=[]
for tile in range(4):
 args=new.suffix_arguments((prefixes[tile],),values[9],scalar(tile))
 out=compiled['prepare_narrow'](*args);jax.block_until_ready(out)
 expected=old.suffix_inputs((prefixes[tile],),scalar(max(0,min(32,33-tile*32))),dense,moe)
 for a,b in zip(out,(expected[0],expected[1],expected[4])):np.testing.assert_array_equal(a,b)
 # Fixture suffix outputs: assembly does not perform model arithmetic.
 suffixes.append((out[0],put(np.zeros((32,8),np.int32)),put(np.zeros((32,8),np.float32)),out[2]))
suffixes=tuple(suffixes)
wide=tuple(jnp.concatenate([s[i] for s in suffixes],axis=2 if i==3 else 0) for i in range(4))
aargs=new.assembly_arguments(prefixes,wide,suffixes)
lowered['assemble']=functions['assemble'].lower(*aargs)
compiled['assemble']=lowered['assemble'].compile()
result=compiled['assemble'](*aargs);jax.block_until_ready(result)
expected=old.assemble_result(prefixes,wide)
for rowresult in result:
 assembled=new.attach_result(rowresult,prefixes[-1])
 for i in new.RESULT_ROW_INDICES:np.testing.assert_array_equal(assembled[i],expected[i])
 for i in (2,3,4):assert assembled[i] is expected[i]
# Same executable, invalid count/tile/offset: safe addresses, latched false health.
for offset,count,tile in ((-1,33,0),(4095,33,0),(2553,-1,0),(2553,129,0),(2553,33,-1),(2553,33,4)):
 p=compiled['prepare_prefix'](rows,scalar(offset),scalar(count),scalar(tile));jax.block_until_ready(p)
 assert not np.asarray(p[0][5]).any() and 0 <= int(p[1]) < 4096
for count,tile in ((-1,0),(129,0),(33,-1),(33,4)):
 p=compiled['prepare_narrow'](prefixes[0][0],prefixes[0][8],scalar(count),scalar(tile));jax.block_until_ready(p)
 assert not np.asarray(p[2]).any()
try:functions['prepare_prefix'].lower(rows,scalar(0),put(np.asarray(33.,np.float32)),scalar(0))
except ValueError:pass
else:raise AssertionError('float count accepted')
for name,program in compiled.items():
 hlo=program.as_text()
 assert not any(k in hlo for k in ('all-reduce(', 'all-gather(', 'collective-permute(', 'convolution(', ' dot(', 'infeed(', 'outfeed('))
 assert program.memory_analysis().alias_size_in_bytes == 0
 memory={key:getattr(program.memory_analysis(),key) for key in new.ALLOCATION_CAPS}
 report=new.inspect_program(name,lowered[name].as_text(),hlo,memory)
 assert report['passed'] and report['physical_collective_count']==0
 print(name,program.memory_analysis(),flush=True)
assert tuple(compiled)==new.PROGRAMS
# Actual abstract preparation -> existing compiler/journal -> raw HLO inspector.
from pathlib import Path
import json,tempfile
from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield import prefill_completed_window_protocol as protocol
from scripts.greenfield.prefill_completed_window_worker import CompletedJournal
from scripts.greenfield.probe_ws32_prefill_layer import compile_program
with tempfile.TemporaryDirectory(prefix='completed-assembly-cpu-') as directory:
 root=Path(directory)
 record=dict(code_hash='a'*40,launch_rank=0,programs={n:{} for n in admission.PROGRAMS})
 identity=dict(protocol=protocol.PROTOCOL,profile=admission.PROFILE,compile_only=False,code_hash='a'*40,launch_rank=0)
 journal=CompletedJournal(root/'compile_journal.jsonl',identity)
 for n,fn,arguments in new.prepare_programs(mesh):
  assert all(isinstance(v,jax.ShapeDtypeStruct) for v in jax.tree.leaves(arguments))
 votes=[]
 def consensus(ok):votes.append(ok);return ok
 helpers=new.compile_programs(mesh=mesh,root=root,record=record,journal=journal,consensus=consensus,compiler=compile_program)
 journal.close()
 assert len(helpers)==4 and len(votes)==9 and all(votes)
 saved=json.loads((root/'runner.json').read_text())
 for name in new.PROGRAMS:
  assert saved['programs'][name]['admission']['passed']
  assert (root/f'{name}.stablehlo.mlir').is_file()
  assert (root/f'{name}.optimized_hlo.txt').is_file()
 rows=[json.loads(line) for line in (root/'compile_journal.jsonl').read_text().splitlines()]
 assert sum(r['stage']=='compiled' for r in rows)==4
 assert sum(r['stage']=='inspected' for r in rows)==4
print('EXPLICIT_ASSEMBLY_CPU32_PASS',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPLICIT_ASSEMBLY_CPU32_PASS" in result.stdout


def test_nine_executable_budget_includes_assembly_and_retained_results():
    from scripts.greenfield import prefill_completed_window_admission as admission
    from scripts.greenfield import prefill_completed_window_assembly as assembly
    from tests.greenfield.hlo.test_prefill_window_admission import memory_inputs
    import pytest

    analyses = {
        name: pin["compiled_memory"]
        for name, pin in admission.registered_programs().items()
    }
    analyses.update(
        {
            name: {
                key: 0 if key == "alias_size_in_bytes" else 1024
                for key in assembly.ALLOCATION_CAPS
            }
            for name in assembly.PROGRAMS
        }
    )
    census, _ = memory_inputs(retained_bytes=50_000_000)
    for active in analyses:
        result = assembly.memory_budget(census, analyses, active_graph=active)
        assert result["estimate_fits"] and not result["numerical_admission"]
        assert result["devices"][0]["resident_code_bytes"] == 25_306_624 + 4 * 1024
        assert (
            result["devices"][0]["estimated_peak_bytes"]
            == 390_000_000
            + 25_306_624
            + 4 * 1024
            + analyses[active]["output_size_in_bytes"]
            + analyses[active]["temp_size_in_bytes"]
        )
    for missing in analyses:
        with pytest.raises(ValueError, match="nine|five model"):
            assembly.memory_budget(
                census,
                {k: v for k, v in analyses.items() if k != missing},
                active_graph="assemble",
            )
    for key, cap in assembly.ALLOCATION_CAPS.items():
        memory = dict(analyses["assemble"])
        memory[key] = cap + 1
        with pytest.raises(ValueError, match="caps"):
            assembly.validate_memory(memory)
        memory[key] = False
        with pytest.raises(ValueError, match="caps"):
            assembly.validate_memory(memory)


def test_assembly_hlo_rejects_model_ops_and_collectives():
    from scripts.greenfield import prefill_completed_window_assembly as assembly
    import pytest

    memory = {
        key: 0 if key == "alias_size_in_bytes" else 1024
        for key in assembly.ALLOCATION_CAPS
    }
    for operation in (
        "dot(a, a)",
        "all-reduce(a), replica_groups={{0,1}}",
        'custom-call(a), custom_call_target="model"',
    ):
        hlo = (
            "HloModule fake\n\nENTRY main {\n a = f32[32,1536] parameter(0)\n ROOT b = f32[32,1536] "
            + operation
            + "\n}\n"
        )
        with pytest.raises(ValueError, match="non-row"):
            assembly.inspect_program("assemble", "stable", hlo, memory)
