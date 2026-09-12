"""Actual producer originals through request replay, fake math/counters only."""
from copy import deepcopy
import gzip
import json
import shutil

import pytest

from scripts.greenfield import ws32_native_benchmark_result as result
from scripts.greenfield import ws32_native_benchmark_memory as memory
from tests.greenfield.validation.test_ws32_native_benchmark_observability import run_case


def test_first_real_host_loop_answer_replayed_and_partial_not_full_quality(tmp_path,monkeypatch):
    (tmp_path/'producer').mkdir()
    made=run_case(tmp_path/'producer',monkeypatch,None)
    root=tmp_path/'collected'; root.mkdir()
    parents=[]
    for rank in range(8):
        directory=root/f'sessions.rank{rank}'/'item000'
        directory.mkdir(parents=True)
        source=made.root/'sessions.rank0'/'item000'
        for name in ('cache.npz','dsa.npz'):
            shutil.copyfile(source/name,directory/name)
        if rank==0:
            for name in ('tokens.jsonl','answer.txt.gz'):shutil.copyfile(source/name,directory/name)
        identity=json.loads((source/'identity.json').read_bytes());identity['rank']=rank
        (directory/'identity.json').write_text(json.dumps(identity))
        slots=[dict(device_id=rank*4+i,slot=rank*4+i) for i in range(4)]
        parents.append(dict(jax_process_index=rank,local_slots=slots,
            programs={n:dict(compiled_memory=m) for n,m in made.memory['compiled_memory'].items()}))
        for phase in ('before_cache','cache_ready','prefill_done'):
            value=json.loads(gzip.decompress((source/(phase+'.json.gz')).read_bytes()))
            value.update(local_slots=slots,process_index=rank)
            for row in value['census']['devices']:row.update(device_id=row['device_id']+rank*4,process_index=rank)
            value['budgets']=memory.budgets(value)
            (directory/(phase+'.json.gz')).write_bytes(gzip.compress(json.dumps(value).encode()))
        final=json.loads((source/'final_memory.json').read_bytes())
        for row in final:row.update(device_id=row['device_id']+rank*4,process_index=rank)
        (directory/'final_memory.json').write_text(json.dumps(final))
        row=json.loads(gzip.decompress((source/'result.json.gz').read_bytes()))
        row['rank']=rank
        if rank:
            row.update(ttft_seconds=None,delivered_request_seconds=None,delivery_boundary='nonoutput_rank')
        trace=row['observations']['trace']
        trace['path']=trace['path'].replace('rank0',f'rank{rank}')
        tracepath=root/trace['path'];tracepath.parent.mkdir(parents=True)
        shutil.copyfile(made.root/trace['path'].replace(f'rank{rank}','rank0'),tracepath)
        (directory/'result.json.gz').write_bytes(gzip.compress(json.dumps(row).encode()))
        (root/f'runner.rank{rank}.json').write_text(json.dumps(dict(code_hash='a'*40,launch_process_id=rank,jax_process_index=rank)))
    args=dict(root=root,payload=made.payload,plan=made.plan,pin='a'*40,parents=parents,
        tokenizer=made.tokenizer,full_index_layers=tuple(range(21)))
    report=result.replay_requests(**args)
    assert report['completed_requests']==1 and report['items'][0]['correct'] is True
    assert report['partial']['index']==1 and len(report['trace_originals'])==8
    assert report['benchmarks']['gpqa_diamond']['correct']==1
    assert report['benchmarks']['gpqa_diamond']['score'] is None
    assert not report['protected_result_sealed'] and not report['matched_card_parity_claim']
    path=root/'sessions.rank7/item000/result.json.gz';original=path.read_bytes()
    for field,value in (('token_ids_sha256','0'*64),('protocol_sha256','0'*64),('ttft_seconds',0.1)):
        row=json.loads(gzip.decompress(original));row[field]=value
        path.write_bytes(gzip.compress(json.dumps(row).encode()))
        with pytest.raises(ValueError):result.replay_requests(**args)
    path.write_bytes(original)
    forged=root/'sessions.rank0/item002/result.json.gz';forged.parent.mkdir()
    forged.write_bytes(original)
    with pytest.raises(ValueError,match='interior'):result.replay_requests(**args)
