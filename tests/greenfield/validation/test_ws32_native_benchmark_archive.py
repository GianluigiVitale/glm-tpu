"""Real SQLite and conditional GCS publisher with fixture owners/model replay.

No TPU, real answer or quality evidence. Faults must preserve DB/originals and
never produce a quality SUCCESS or require re-running model requests.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from scripts.greenfield import ws32_native_benchmark_archive as archive
from tests.greenfield.validation.test_ws32_native_benchmark_database import fixture, TAG, PIN
from tests.greenfield.validation.test_ws32_native_benchmark_launch import fleet
from tests.greenfield.validation.test_ws32_dense_frontier_transport import Bucket


@pytest.fixture
def case(tmp_path):
    root=tmp_path/TAG;root.mkdir()
    values=fixture(root)
    values['report'].update(cold={'fixture_only':True},trace_physical_coverage_verified=True)
    original=fleet()
    census=''.join('FP8_IDLE '+json.dumps(dict(host=r['host'],boot_id=r['boot_id'],devices=[0,1,2,3]))+'\n'
                   for r in original)
    for name in ('census_pre.txt','census_post.txt'):(root/name).write_text(census)
    for name,value in (('launch.json',dict(tag=TAG,code_hash=PIN)),('requests.json',values['payload']),
                       ('protocol.json',values['plan']),('request_replay.json',values['report']),
                       ('collection.json',dict(complete=True))):
        (root/name).write_text(json.dumps(value))
    for name in ('sync.txt','prepared.txt'):(root/name).write_text('fixture eight hosts')
    for rank,r in enumerate(original):
        p=r['processes'][0]
        owner=dict(p,hostname=r['host'],boot_id=r['boot_id'])
        (root/'collected'/f'runner.rank{rank}.json').write_text(json.dumps(dict(
            code_hash=PIN,launch_process_id=rank,owner=owner)))
    idle=deepcopy(original)
    for r in idle:r['processes']=[]
    (root/'native_watch.jsonl').write_text(''.join(json.dumps(dict(status='OBSERVED',tag=TAG,pin=PIN,
        fleet=f))+'\n' for f in (original,idle,idle)))
    bucket=Bucket();cold_receipts=[];request_receipts=[]
    # Manifests represent prior collector replay; test their exact remote union
    # here using the real conditional publisher/readback, not invented receipts.
    for rank in range(8):
        for channel,receipts in ((archive.cold,cold_receipts),(archive.requests,request_receipts)):
            prefix=channel.prefix(TAG,rank)
            source=root/'fixture_original.json';source.write_text(json.dumps(dict(rank=rank)))
            child=archive.publish_exact(bucket,prefix+'original.json',source,archive.digest_file(source),compressed=False)
            manifest=dict(rank=rank,tag=TAG,code_hash=PIN,files=[child])
            source.write_text(json.dumps(manifest))
            archive.publish_exact(bucket,prefix+'manifest.json',source,archive.digest_file(source),compressed=False)
            receipts.append(dict(manifest=manifest) if channel is archive.cold else manifest)
    kwargs=dict(root=root,tag=TAG,pin=PIN,report=values['report'],cold_receipts=cold_receipts,
        request_receipts=request_receipts,blobs={n:bucket.get_blob(n) for n in bucket.objects},
        client=bucket.client(),db_path=values['database'])
    return NS(root=root,bucket=bucket,kwargs=kwargs)


def test_actual_db_archive_repeat_no_quality_success_or_full_database_copy(case):
    result=archive.archive(**case.kwargs)
    assert result['completed_requests']==1 and result['evidence_archived']
    assert not result['quality_pass_claim'] and not result['project_complete']
    assert result['benchmarks']['gpqa_diamond']['score'] is None
    before=deepcopy(case.bucket.objects)
    assert archive.archive(**case.kwargs)==result
    assert before==case.bucket.objects
    assert not any(n.endswith('/SUCCESS') or n.endswith('results.db') for n in before)
    assert f'results/{TAG}/EVIDENCE_ARCHIVED.json' in before


def test_archive_failure_retries_existing_db_without_duplicate_or_model_rerun(case,monkeypatch):
    original=archive.publish_exact
    def fail(bucket,name,*args,**kwargs):
        if name.endswith('db_link.json'):raise RuntimeError('injected archive interruption')
        return original(bucket,name,*args,**kwargs)
    monkeypatch.setattr(archive,'publish_exact',fail)
    with pytest.raises(RuntimeError,match='interruption'):archive.archive(**case.kwargs)
    assert not any(n.endswith('EVIDENCE_ARCHIVED.json') for n in case.bucket.objects)
    monkeypatch.setattr(archive,'publish_exact',original)
    result=archive.archive(**case.kwargs)
    connection=archive.database.provenance.connect(str(case.kwargs['db_path']))
    assert connection.execute('SELECT count(*) FROM runs').fetchone()==(1,)
    assert connection.execute('SELECT count(*) FROM items').fetchone()==(1,)
    assert result['run_id']==1
    connection.close()


@pytest.mark.parametrize('fault',['owner','missing_pid','busy','generation','extra','region','trace','rank','census_boot'])
def test_refusal_before_database_or_terminal(case,fault):
    if fault=='owner':
        p=case.root/'collected/runner.rank3.json';r=json.loads(p.read_text());r['owner']['start_ticks']='changed';p.write_text(json.dumps(r))
    elif fault in ('missing_pid','busy'):
        p=case.root/'native_watch.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
        if fault=='missing_pid':rows=rows[1:]
        else:rows[-1]['fleet'][0]['holders']=[999]
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif fault=='generation':
        name=next(iter(case.bucket.objects));gen,data=case.bucket.objects[name];case.bucket.objects[name]=(gen+100,data)
    elif fault=='extra':
        name=archive.cold.prefix(TAG,0)+'extra';case.bucket.objects[name]=(1,b'bad');case.kwargs['blobs'][name]=case.bucket.get_blob(name)
    elif fault=='region':case.bucket.location='EU'
    elif fault=='trace':case.kwargs['report']['trace_physical_coverage_verified']=False
    elif fault=='census_boot':
        p=case.root/'census_post.txt';p.write_text(p.read_text().replace('boot3','newboot3'))
    else:case.kwargs['cold_receipts'][3]['manifest']['rank']=4
    with pytest.raises(ValueError):archive.archive(**case.kwargs)
    assert not case.kwargs['db_path'].exists()
    assert not any(n.endswith('EVIDENCE_ARCHIVED.json') for n in case.bucket.objects)
