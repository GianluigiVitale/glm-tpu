"""A partial or mixed fleet must never become a complete timing receipt."""
import json
import copy

import pytest

from tools.perf_tpu_microbench import summarize, validate_global_max_benchmark


def test_summary_requires_complete_consistent_fleet(tmp_path):
    def write(rank, **changes):
        row = dict(rank=rank, hostname=f"host-{rank}", devices=32, which="attention",
                   jax="test", finished_utc="test", source_sha256={"source.py": "abc"})
        row.update(changes)
        (tmp_path / f"microbench.rank{rank}.json").write_text(json.dumps(row))

    for rank in range(7):
        write(rank)
    with pytest.raises(ValueError, match="eight"):
        summarize(tmp_path)
    write(7, finished_utc=None)
    with pytest.raises(ValueError, match="completed"):
        summarize(tmp_path)
    write(7, source_sha256={"source.py": "different"})
    with pytest.raises(ValueError, match="fingerprints"):
        summarize(tmp_path)
    write(7)
    assert summarize(tmp_path)["ranks"] == list(range(8))
    assert len(summarize(tmp_path)['originals_sha256'])==8


def test_summary_refuses_missing_nested_metric(tmp_path):
    for rank in range(8):
        row=dict(rank=rank,hostname=f'host-{rank}',devices=32,which='empty_routes',jax='test',
                 finished_utc='test',source_sha256={'source.py':'abc'},
                 result=dict(empty=dict(nonzero=0,finite=True)))
        if rank==7:del row['result']['empty']['finite']
        (tmp_path/f'microbench.rank{rank}.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='metric fields'):
        summarize(tmp_path)


def global_max_fixture():
    ranks=[]
    for rank in range(8):
        memory=[dict(device_id=rank*4+i,bytes_in_use=10,peak_bytes_in_use=20,bytes_limit=100)
                for i in range(4)]
        planned=[dict(device_id=m['device_id'],predicted_bytes=30,limit=100,fits=True) for m in memory]
        value=dict(hlo_admission=dict(passed=True,num_partitions=32),hlo_sha256='a'*64,
            fleet_graph_consensus=True,fleet_health_passed=True,max_abs_vs_frozen=0.,
            timing=dict(samples=5,min_ms=1.,p50_ms=2.,p90_ms=3.,p99_ms=4.,mean_ms=2.5),
            memory_admission=dict(passed=True,chips=planned),memory_after=memory)
        cases={f'rows{n}_{pattern}':dict(rows=n,selected_count=count,
            modes={mode:copy.deepcopy(value) for mode in ('frozen','global_max')})
            for n in (1,2,3,4,32) for pattern,count in
            (('prefix64',64),('balanced2048',2048),('concentrated1024',1024))}
        ranks.append(dict(rank=rank,hostname=f'host-{rank}',devices=32,which='global_max_attention',
            jax='test',finished_utc='test',source_sha256={'source.py':'abc'},iters=5,
            global_max_attention=dict(context_capacity=8192,numerical_exactness=False,cases=cases)))
    return ranks


def test_global_max_negative_numerical_result_stays_visible(tmp_path):
    ranks=global_max_fixture()
    ranks[-1]['global_max_attention']['cases']['rows3_balanced2048']['modes']['global_max']['max_abs_vs_frozen']=.5
    for rank in ranks:
        (tmp_path/f'microbench.rank{rank["rank"]}.json').write_text(json.dumps(rank))
    report=summarize(tmp_path)
    assert report['global_max_attention']['cases']['rows3_balanced2048']['modes']['global_max']['max_abs_vs_frozen']==dict(min=0.,max=.5)


@pytest.mark.parametrize('bad',['cases','mode','rows','count','health','graph','hash','nan_error',
    'reference_error','samples','nan_time','zero_time','ordering','memory','chip','plan','iters','owner'])
def test_global_max_summary_refuses_incomplete_comparison(bad):
    ranks=global_max_fixture();r=ranks[-1];case=r['global_max_attention']['cases']['rows3_balanced2048']
    v=case['modes']['global_max']
    if bad=='cases':del r['global_max_attention']['cases']['rows32_prefix64']
    if bad=='mode':del case['modes']['frozen']
    if bad=='rows':case['rows']=2
    if bad=='count':case['selected_count']=128
    if bad=='health':v['fleet_health_passed']=False
    if bad=='graph':v['hlo_admission']['passed']=False
    if bad=='hash':v['hlo_sha256']='b'*64
    if bad=='nan_error':v['max_abs_vs_frozen']=float('nan')
    if bad=='reference_error':case['modes']['frozen']['max_abs_vs_frozen']=.5
    if bad=='samples':v['timing']['samples']=4
    if bad=='nan_time':v['timing']['p50_ms']=float('nan')
    if bad=='zero_time':v['timing']['min_ms']=0
    if bad=='ordering':v['timing']['p90_ms']=1
    if bad=='memory':v['memory_after'].pop()
    if bad=='chip':v['memory_after'][0]['device_id']=0
    if bad=='plan':v['memory_admission']['chips'][0]['fits']=False
    if bad=='iters':r['iters']=6
    if bad=='owner':
        a=ranks[0]['global_max_attention']['cases']['rows3_balanced2048']['modes']['global_max']
        for field in ('memory_after','memory_admission'):
            a[field],v[field]=v[field],a[field]
    with pytest.raises(ValueError):validate_global_max_benchmark(ranks)


def test_synthetic_generator_respects_partial_replication_cpu32():
    import os
    import subprocess
    import sys

    code = r'''
import jax, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from tools.perf_tpu_microbench import _sharded_random
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
for spec in (P('expert'), P(None,'feature'), P('expert','feature'), P()):
    a=_sharded_random(mesh,(8,128),spec,'bf16',1)
    groups={}
    for shard in a.addressable_shards:
        groups.setdefault(str(shard.index),[]).append(np.asarray(shard.data))
    for values in groups.values():
        for other in values[1:]: np.testing.assert_array_equal(values[0],other)
    assert len({values[0].tobytes() for values in groups.values()})==len(groups)
print('all replicated axes agree; partitioned shards differ')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
