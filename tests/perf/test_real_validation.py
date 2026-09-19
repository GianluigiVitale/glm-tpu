"""Fail-closed research HLO and physical-memory admission, CPU only."""
import pytest

from glm_tpu.perf.real_validation import inspect_research_hlo, memory_projection


def hlo(*, groups=None, dtype='f32', count=4, partitions=32, opcode='all-reduce'):
    if groups is None:
        groups = '{'+','.join('{'+','.join(map(str,range(i,i+4)))+'}' for i in range(0,32,4))+'}'
    return f'''HloModule test, num_partitions={partitions}
ENTRY main {{
  p = {dtype}[{count}] parameter(0)
  ROOT r = {dtype}[{count}] {opcode}(p), replica_groups={groups}, use_global_device_ids=true
}}
'''


def test_axis_exchange_admitted_and_censused():
    r=inspect_research_hlo(hlo())
    assert r['passed'] and r['maximum_collective_payload_bytes']==16
    assert r['collectives']=={'all-reduce':1}
    groups='{'+','.join('{'+','.join(map(str,range(i,32,4)))+'}' for i in range(4))+'}'
    assert inspect_research_hlo(hlo(groups=groups,opcode='all-gather'))['passed']


@pytest.mark.parametrize('options',[
    {'partitions':8}, {'groups':'{{0,1,2,3}}'}, {'dtype':'f8e4m3fn'},
    {'count':128*1024**2}, {'opcode':'all-to-all'},
    {'groups':'{{'+','.join(map(str,range(32)))+'}}','count':2048},
    {'groups':'{{'+','.join(map(str,range(32)))+'},{0}}'},
])
def test_graph_refusals(options):
    with pytest.raises(ValueError):inspect_research_hlo(hlo(**options))


def test_partition_local_group_ids_are_refused():
    with pytest.raises(ValueError):
        inspect_research_hlo(hlo().replace('use_global_device_ids=true','use_global_device_ids=false'))


def test_async_exchange_payload_is_checked():
    assert inspect_research_hlo(hlo(opcode='all-reduce-start'))['passed']
    with pytest.raises(ValueError):inspect_research_hlo(hlo(opcode='all-reduce-start',count=128*1024**2))


def test_memory_all_four_chips_and_resident_output_scratch_code():
    stats=[dict(device_id=i,bytes_in_use=100,bytes_limit=1000) for i in range(4)]
    mem=dict(output_size_in_bytes=200,temp_size_in_bytes=300,generated_code_size_in_bytes=50,alias_size_in_bytes=100)
    r=memory_projection(stats,mem,reserve_bytes=100)
    assert r['passed'] and {x['predicted_bytes'] for x in r['chips']}=={650}
    stats[3]['bytes_limit']=650
    assert not memory_projection(stats,mem,reserve_bytes=100)['passed']
    with pytest.raises(ValueError):memory_projection(stats[:3],mem)
    stats[3]['device_id']=0
    with pytest.raises(ValueError):memory_projection(stats,mem)
    with pytest.raises(ValueError):memory_projection(stats,dict(mem,temp_size_in_bytes=-1))
