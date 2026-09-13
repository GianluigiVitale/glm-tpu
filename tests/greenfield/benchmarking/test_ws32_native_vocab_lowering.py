"""Native-only physical interface mutations; not a compiler-value proof."""
from collections import Counter

import pytest

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_collective_hlo import _physical_records
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import (
    _expected, _nucleus_exchange,
)


HLO = """HloModule native_vocab_interface, num_partitions=32

%sum (x: bf16[], y: bf16[]) -> bf16[] {
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %add = bf16[] add(%x, %y)
}

ENTRY %main (fragment: bf16[19360]) -> bf16[154880] {
  %fragment = bf16[19360] parameter(0)
  %rank = u32[] partition-id()
  %table = u32[32] constant({...})
  %lookup = u32[1] dynamic-slice(%table, %rank), dynamic_slice_sizes={1}
  %stride = u32[1] constant({19360})
  %product = u32[1] multiply(%lookup, %stride)
  %offset = u32[] bitcast(%product)
  %zero = bf16[] constant(0)
  %base = bf16[154880] broadcast(%zero), dimensions={}
  %insert = bf16[154880] dynamic-update-slice(%base, %fragment, %offset)
  ROOT %exchange = bf16[154880] all-reduce(%insert), channel_id=1, replica_groups={{0,4,8,12,16,20,24,28},{1,5,9,13,17,21,25,29},{2,6,10,14,18,22,26,30},{3,7,11,15,19,23,27,31}}, use_global_device_ids=true, to_apply=%sum
}
"""


def inspect(text=HLO):
    index = PrefillHloIndex(parse_hlo_module(text))
    records, _ = _physical_records(index, index.module.instructions)
    return _nucleus_exchange(index, records)


def test_zero_padded_exchange_interface_has_explicit_scope():
    report = inspect()
    assert report['lowered'] is True
    assert report['form'] == 'zero_padded_expert8_all_reduce'
    assert 'OPAQUE_PARTITION_TABLE_VALUES' in report['not_proven']
    assert 'FRAGMENT_VALUES' in report['not_proven']


@pytest.mark.parametrize('before,after', [
    ('constant(0)', 'constant(1)'),
    ('constant({19360})', 'constant({19359})'),
    ('u32[32] constant', 'u32[31] constant'),
    ('u32[] bitcast(%product)', 's32[] bitcast(%product)'),
    ('multiply(%lookup, %stride)', 'add(%lookup, %stride)'),
    ('bf16[19360] parameter(0)', 'bf16[19359] parameter(0)'),
    ('dynamic-update-slice(%base, %fragment, %offset)', 'copy(%base)'),
    ('use_global_device_ids=true', 'use_global_device_ids=false'),
    ('{0,4,8,12,16,20,24,28}', '{1,4,8,12,16,20,24,28}'),
    ('bf16[] add(%x, %y)', 'bf16[] maximum(%x, %y)'),
])
def test_malformed_or_nonlocal_lowering_refuses(before, after):
    assert before in HLO
    with pytest.raises((ValueError, KeyError, IndexError)):
        inspect(HLO.replace(before, after))


@pytest.mark.parametrize('rows', [114, 128])
def test_lowering_changes_only_one_native_head_physical_record(rows):
    gathered = _expected(rows, canonical_dense=True, nucleus_head=True)
    lowered = _expected(rows, canonical_dense=True, nucleus_head=True,
                        nucleus_gather_lowered=True)
    assert sum(gathered.values()) == sum(lowered.values()) == 787
    assert sum((gathered-lowered).values()) == 1
    shape = (('bf16', (154880,)),)
    assert lowered-gathered == Counter({
        ('outer',-1,'all-reduce','expert','add',-1,shape,shape): 1})
    with pytest.raises(ValueError, match='explicit nucleus'):
        _expected(rows, nucleus_gather_lowered=True)


def test_no_lowered_candidate_keeps_original_gather_schedule():
    assert _nucleus_exchange(None, []) == dict(form='all_gather',lowered=False)
