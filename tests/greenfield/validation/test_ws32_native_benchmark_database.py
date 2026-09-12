"""Real SQLite transactions and original output rows; no model/quality proof."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json

import pytest

from scripts.greenfield import ws32_native_benchmark_database as db
from scripts.greenfield import ws32_native_benchmark_protocol as protocol
from tests.greenfield.validation.test_ws32_native_benchmark_requests import capsule

TAG='greenfield_ws32_native_benchmark_20260912T220000000000000Z'
PIN='a'*40


def fixture(tmp_path, *, complete=False):
    payload,plan=capsule()
    for request in payload['requests']:request['item']['prompt']='fixture question, not real dataset'
    plan=protocol.protocol(payload,plan['tokenizer_files'])
    count=228 if complete else 1
    items=[]
    for i,request in enumerate(payload['requests'][:count]):
        correct=i%2==0 if i<198 else None
        item=dict(index=i,request_id=request['request_id'],dataset=request['dataset'],
            correct=correct,extracted='A' if correct else None,generated_tokens=2,token_ids_sha256='b'*64)
        items.append(item)
        path=tmp_path/'collected'/'sessions.rank0'/f'item{i:03d}';path.mkdir(parents=True)
        row=dict(item,asked_utc=datetime.now(timezone.utc).isoformat(),delivered_request_seconds=2.0,
            finish_reason='eos' if i%3 else 'length')
        (path/'result.json.gz').write_bytes(gzip.compress(json.dumps(row).encode()))
        (path/'answer.txt.gz').write_bytes(gzip.compress(b'fixture reasoning</think>A'))
    report=dict(schema='ws32_native_request_replay_v1',completed_requests=count,items=items,
        matched_card_parity_claim=False,protected_result_sealed=False,benchmarks={})
    for name in protocol.COUNTS:
        report['benchmarks'][name]=dict(score=0.5 if complete and name=='gpqa_diamond' else None,
            status='FULL_SET_SCORED_PROTOCOL_CAVEATS_APPLY' if complete and name=='gpqa_diamond' else 'INCOMPLETE_OR_UNSCORED')
    return dict(database=tmp_path/'results.db',root=tmp_path,tag=TAG,pin=PIN,payload=payload,plan=plan,report=report)


@pytest.mark.parametrize('complete',[False,True])
def test_real_db_commit_replay_no_completed_subset_accuracy_or_fake_math_score(tmp_path,complete):
    kwargs=fixture(tmp_path,complete=complete)
    value=db.record_result(**kwargs)
    assert value==db.record_result(**kwargs)
    assert len(value['rows']['runs'])==1
    assert len(value['rows']['items'])==(228 if complete else 1)
    assert value['rows']['runs'][0]['model_revision'] is None
    summaries={r['benchmark']:r for r in value['rows']['summary']}
    assert summaries['gpqa_diamond']['value']==(50.0 if complete else None)
    assert summaries['gpqa_diamond']['card_value']==pytest.approx(91.2)
    assert summaries['aime_2026']['value'] is None
    if complete:
        assert all(r['correct'] is None for r in value['rows']['items'][198:])
    altered=deepcopy(kwargs);altered['report']['changed']=True
    with pytest.raises(ValueError,match='different originals'):db.record_result(**altered)


def test_item_failure_rolls_back_entire_new_run_preserving_existing_history(tmp_path,monkeypatch):
    kwargs=fixture(tmp_path,complete=True)
    connection=db.provenance.connect(str(kwargs['database']))
    old=db.provenance.start_run(connection,model='existing history',fork_repo=str(db.REPO))
    connection.close()
    original=db.provenance.record_item
    count=0
    def fail(connection,run_id,**row):
        nonlocal count
        original(connection,run_id,**row);count+=1
        if count==3:raise RuntimeError('injected after third insert')
    monkeypatch.setattr(db.provenance,'record_item',fail)
    with pytest.raises(RuntimeError,match='third insert'):db.record_result(**kwargs)
    connection=db.provenance.connect(str(kwargs['database']))
    assert connection.execute('SELECT run_id FROM runs').fetchall()==[(old,)]
    assert connection.execute('SELECT count(*) FROM items').fetchone()==(0,)
    assert connection.execute('SELECT count(*) FROM native_benchmark_links').fetchone()==(0,)
    connection.close()


def test_modified_summary_detected_on_idempotent_replay(tmp_path):
    kwargs=fixture(tmp_path)
    value=db.record_result(**kwargs)
    connection=db.provenance.connect(str(kwargs['database']))
    connection.execute('UPDATE summary SET value=100 WHERE run_id=?',(value['run_id'],))
    connection.commit();connection.close()
    with pytest.raises(ValueError,match='rows changed'):db.record_result(**kwargs)
