"""Atomic, idempotent native benchmark linkage using existing provenance schema.

No legacy model execution. Store every replayed item, including misses/length
stops and unjudged AIME. Never call provenance.finalize on a partial set: it
would compute completed-only accuracy. This is DB linkage, not a quality PASS.
The protected outer caller first authenticates originals, replay and cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any

from bench import provenance
from scripts.greenfield.ws32_native_benchmark_protocol import REPO, canonical, validate
from scripts.greenfield.ws32_native_benchmark_result import read
from scripts.greenfield.ws32_native_benchmark_transport import _identity
from scripts.greenfield.ws32_history_preflight import _plain_path

PRIMARY_DB = Path('/home/gianl/glm-tpu/bench/results.db')
LEDGER_SCHEMA = '''CREATE TABLE IF NOT EXISTS native_benchmark_links (
    tag TEXT PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(run_id),
    code_hash TEXT NOT NULL, report_sha256 TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
    rows_sha256 TEXT NOT NULL
)'''


class Transaction:
    """Let old per-row helpers participate in ONE outer SQLite transaction."""
    def __init__(self, connection: sqlite3.Connection) -> None: self.connection=connection
    def execute(self, *args: Any) -> Any: return self.connection.execute(*args)
    def commit(self) -> None: pass


def export_rows(connection: sqlite3.Connection, run_id: int) -> dict:
    result={}
    for table, order in (('runs','run_id'),('items','id'),('summary','id')):
        cursor=connection.execute(f'SELECT * FROM {table} WHERE run_id=? ORDER BY {order}',(run_id,))
        names=[c[0] for c in cursor.description]
        result[table]=[dict(zip(names,row,strict=True)) for row in cursor]
    return result


def record_result(*, database: Path, root: Path, tag: str, pin: str,
                  payload: dict, plan: dict, report: dict) -> dict:
    _identity(tag,pin,0); validate(payload,plan)
    _plain_path(database)
    if (report.get('schema')!='ws32_native_request_replay_v1'
            or report.get('matched_card_parity_claim') is not False
            or report.get('protected_result_sealed') is not False
            or len(report['items'])!=report['completed_requests']
            or report['completed_requests']>len(payload['requests'])):
        raise ValueError('native database requires original replay scope')
    digest=sha256(canonical(report)).hexdigest()
    payload_digest=sha256(canonical(payload)).hexdigest()
    rows=[]
    for index,item in enumerate(report['items']):
        request=payload['requests'][index]
        directory=root/'collected'/'sessions.rank0'/f'item{index:03d}'
        row=json.loads(read(directory/'result.json.gz',4<<20,compressed=True))
        if (item['index']!=index or item['request_id']!=request['request_id']
                or item['dataset']!=request['dataset'] or row['request_id']!=item['request_id']
                or row['correct'] is not item['correct'] or row['extracted']!=item['extracted']
                or row['token_ids_sha256']!=item['token_ids_sha256']
                or row['generated_tokens']!=item['generated_tokens']):
            raise ValueError('database item differs from replayed original')
        asked=datetime.fromisoformat(row['asked_utc'])
        if asked.tzinfo is None:raise ValueError('actual request UTC missing timezone')
        text=read(directory/'answer.txt.gz',16<<20,compressed=True).decode()
        prompt=canonical(dict(prompt_ids=request['prompt_ids'],
            prompt_ids_sha256=request['prompt_ids_sha256'],
            user_prompt=request['item']['prompt'],system_prompt=request['item'].get('system_prompt'))).decode()
        rows.append(dict(benchmark=request['dataset'],item_id=request['item']['item_id'],
            prompt=prompt,gold=request['item']['gold'],raw_output=text,extracted=item['extracted'],
            correct=item['correct'],score=None if item['correct'] is None else float(item['correct']),
            n_prompt_tokens=len(request['prompt_ids']),n_gen_tokens=item['generated_tokens'],
            latency_ms=row['delivered_request_seconds']*1000,seed=request['seed'],asked_utc=row['asked_utc'],
            finish_reason=row['finish_reason'],truncated=row['finish_reason']=='length'))
    connection=provenance.connect(str(database))
    try:
        connection.execute(LEDGER_SCHEMA); connection.commit()
        connection.execute('BEGIN IMMEDIATE')
        existing=connection.execute('SELECT run_id,code_hash,report_sha256,payload_sha256,rows_sha256 FROM native_benchmark_links WHERE tag=?',(tag,)).fetchone()
        if existing is not None:
            run_id,old_pin,old_digest,old_payload,old_rows=existing
            if (old_pin,old_digest,old_payload)!=(pin,digest,payload_digest):
                raise ValueError('native tag already linked to different originals; no overwrite')
            exported=export_rows(connection,run_id)
            if sha256(canonical(exported)).hexdigest()!=old_rows:
                raise ValueError('linked database rows changed')
            # Detect deletion or changes after initial transaction, not only a
            # surviving idempotency row. Compare original helper columns.
            if len(exported['items'])!=len(rows):raise ValueError('linked database item count changed')
            for stored,expected in zip(exported['items'],rows,strict=True):
                if any(stored[k]!=v for k,v in expected.items()):
                    raise ValueError('linked database item changed')
        else:
            tx=Transaction(connection)
            # A pinned model-card revision is not automatically the weight
            # checkpoint revision. Weight identity stays in the cold originals.
            run_id=provenance.start_run(tx,model='zai-org/GLM-5.2-FP8',revision=None,
                env=dict(native=True,run_tag=tag,code_hash=pin,protocol=plan,report_sha256=digest,
                         no_legacy_execution=True,delivered_boundary='rank0_local_jsonl_write_flush'),
                pod='db-v4-64-od',note='Native original request replay; task quality may be INCOMPLETE. '+tag,
                harness_repo=str(REPO),fork_repo=str(REPO))
            for row in rows:provenance.record_item(tx,run_id,**row)
            utc=datetime.now(timezone.utc).isoformat()
            for name,summary in report['benchmarks'].items():
                subset=[r for r in rows if r['benchmark']==name]
                scored=[r for r in subset if r['correct'] is not None]
                full=len(scored)==plan['counts'][name]
                value=100.0*sum(r['correct'] for r in scored)/len(scored) if full else None
                expected=summary['score']
                if ((value is None)!=(expected is None)
                        or value is not None and abs(value-100*expected)>1e-10):
                    raise ValueError('full-set database score differs from replay')
                card=100.0*plan['targets'][name]
                note=canonical(dict(status=summary['status'],registered=plan['counts'][name],
                    completed=len(subset),scored=len(scored),n_truncated=sum(r['truncated'] for r in subset),
                    protocol_caveats=plan.get('caveats',plan.get('protocol_caveats',[])),
                    matched_card_parity_claim=False)).decode()
                connection.execute('INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,card_value,delta,note) VALUES (?,?,?,?,?,?,?,?,?)',
                    (run_id,name,utc,len(scored),'acc',value,card,None if value is None else value-card,note))
            exported=export_rows(connection,run_id)
            connection.execute('INSERT INTO native_benchmark_links VALUES (?,?,?,?,?,?)',
                (tag,run_id,pin,digest,payload_digest,sha256(canonical(exported)).hexdigest()))
        connection.commit()
        return dict(schema='ws32_native_database_link_v1',run_id=run_id,tag=tag,code_hash=pin,
            report_sha256=digest,payload_sha256=payload_digest,rows=exported,
            rows_sha256=sha256(canonical(exported)).hexdigest(),quality_pass_claim=False)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
