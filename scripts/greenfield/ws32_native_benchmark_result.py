"""Replay original native request bytes and report full-set card score gaps.

Caller authenticates the generation-bound transport, cold fleet and topology.
No result here creates SUCCESS or substitutes partial-set quality for a full
benchmark. AIME remains unscored until the owner approves the required judge.
"""
from __future__ import annotations

import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

import ml_dtypes
import numpy as np

from scripts.greenfield import ws32_native_benchmark_protocol as protocol
from scripts.greenfield import ws32_native_benchmark_memory as memory
from scripts.greenfield.ws32_native_benchmark_requests import score_answer
from scripts.greenfield.ws32_delivery_phase_evidence import _advance, _boundary
from scripts.greenfield.ws32_native_benchmark_observability import WITNESS_CAP, TRACE_CAP
from scripts.greenfield.ws32_history_preflight import _plain_path
from glm_tpu.greenfield.validation.ws32_short_context import (
    compare_ws32_dsa_within_engine, validate_ws32_cache_probe,
)


def read(path: Path, cap: int, *, compressed: bool = False) -> bytes:
    _plain_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError("native result original missing or oversized: "+str(path))
    raw = path.read_bytes()
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream: raw = stream.read(cap+1)
    if len(raw) > cap: raise ValueError("native result inflated original exceeds cap")
    return raw


def arrays(path: Path, names: set[str]) -> dict:
    raw=read(path,WITNESS_CAP)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries=z.infolist()
        if (len(entries)!=len(names) or {i.filename for i in entries}!={n+'.npy' for n in names}
                or sum(i.file_size for i in entries)>WITNESS_CAP):
            raise ValueError("native compact NPZ schema/unpacked size differs")
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        return {n:archive[n] for n in names}


def replay_observation(directory: Path, row: dict, *, prompt_tokens: int,
                       parent: dict, full_index_layers: tuple[int,...]) -> dict:
    slots={r["device_id"]:r["slot"] for r in parent["local_slots"]}
    process=parent["jax_process_index"]
    previous=None
    for phase in ("before_cache","cache_ready","prefill_done"):
        record=json.loads(read(directory/(phase+'.json.gz'),4<<20,compressed=True))
        if (record["phase"]!=phase or record["local_slots"]!=parent["local_slots"]
                or record["process_index"]!=process
                or record["compiled_memory"]!={n:r["compiled_memory"] for n,r in parent["programs"].items()}):
            raise ValueError("request memory phase/owner/compiled identity differs")
        memory.validate_record(record)
        current=_boundary(record["census"]["devices"],slots,process,nested=True)
        if previous is not None:_advance(previous,current)
        previous=current
    final=json.loads(read(directory/'final_memory.json',64<<10))
    current=_boundary(final,slots,process,nested=False)
    _advance(previous,current)
    cache=arrays(directory/'cache.npz',{'position','kv_bfloat16_bits','index_bfloat16_bits','contract_valid'})
    if any(cache[n].dtype!=np.uint16 for n in ('kv_bfloat16_bits','index_bfloat16_bits')):
        raise ValueError("cache witness must retain original BF16 bits")
    checked=validate_ws32_cache_probe(position=cache['position'],
        kv_rows=cache['kv_bfloat16_bits'].view(ml_dtypes.bfloat16),
        index_rows=cache['index_bfloat16_bits'].view(ml_dtypes.bfloat16),contract_valid=cache['contract_valid'],
        expected_position=prompt_tokens+row['generated_tokens']-2,num_layers=78,
        full_indexer_count=len(full_index_layers),packed_cache_width=640,index_width=128)
    observed=row['observations']
    if not checked['passed'] or checked!=observed['cache']:
        raise ValueError("original final cache replay failed")
    if row['generated_tokens']>1:
        dsa=arrays(directory/'dsa.npz',{'producer_layer_ids','selected_positions','selected_valid_counts','selected_scores'})
        checked=compare_ws32_dsa_within_engine(**dsa,decode_position=prompt_tokens,step=0,
            expected_producer_layer_ids=np.asarray(full_index_layers,np.int32))
        if (not checked['passed'] or checked!=observed['dsa'] or observed['dsa_observed'] is not True
                or dsa['selected_positions'].shape!=(len(full_index_layers),1,2048)
                or not np.all(dsa['selected_valid_counts']==min(prompt_tokens+1,2048))
                or observed['instrumented_decode_indices']!=[0]):
            raise ValueError("original executing DSA replay failed")
    elif observed['instrumented_decode_indices'] or observed['dsa_observed']:
        raise ValueError("prefill-only response claims a decode observation")
    return dict(peak=max(r['peak_bytes_in_use'] for r in final),trace=observed['trace'])


def replay_requests(*, root: Path, payload: dict, plan: dict, pin: str,
                    parents: list[dict], tokenizer: Any, full_index_layers: tuple[int,...]) -> dict:
    protocol.validate(payload,plan)
    if len(parents)!=8:raise ValueError("native request replay needs eight cold parents")
    registry=protocol.benchmark_registry()
    results=[];traces=[];maximum_peak=0;partial=None
    for index,request in enumerate(payload['requests']):
        directories=[root/f'sessions.rank{r}'/f'item{index:03d}' for r in range(8)]
        present=[(d/'result.json.gz').is_file() for d in directories]
        if not all(present):
            partial=dict(index=index,completed_ranks=[r for r,p in enumerate(present) if p],
                explanation='Not a complete scored fleet request; retain raw partials and failures')
            if any((root/f'sessions.rank{r}'/f'item{j:03d}'/'result.json.gz').exists()
                   for r in range(8) for j in range(index+1,len(payload['requests']))):
                raise ValueError("native request sequence has a missing interior item")
            break
        rows=[json.loads(read(d/'result.json.gz',4<<20,compressed=True)) for d in directories]
        raw=read(directories[0]/'tokens.jsonl',32<<20)
        if not raw.endswith(b'\n'):raise ValueError("completed token stream has a partial event")
        events=[json.loads(line) for line in raw.splitlines()]
        ids=[]
        for i,event in enumerate(events):
            if (set(event)!={'request_id','index','token_id','finish_reason'}
                    or event['request_id']!=request['request_id'] or type(event['index']) is not int or event['index']!=i
                    or type(event['token_id']) is not int or not 0<=event['token_id']<plan['vocab_size']):
                raise ValueError("raw token delivery frontier differs")
            expected=('eos' if event['token_id'] in plan['eos_ids'] else
                      'length' if i+1==plan['max_new_tokens'] else None)
            if event['finish_reason']!=expected or (i<len(events)-1 and expected is not None):
                raise ValueError("raw token stop policy differs")
            ids.append(event['token_id'])
        if not ids or len(ids)>plan['max_new_tokens'] or events[-1]['finish_reason'] is None:
            raise ValueError("completed answer lacks a policy-consistent terminal token")
        digest=sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest()
        for rank,(row,directory,parent) in enumerate(zip(rows,directories,parents,strict=True)):
            outer=json.loads(read(root/f'runner.rank{rank}.json',2<<20))
            if outer['code_hash']!=pin or outer['launch_process_id']!=rank or outer['jax_process_index']!=parent['jax_process_index']:
                raise ValueError("request worker source/owner differs")
            identity=json.loads(read(directory/'identity.json',4096))
            if identity!={'index':index,'request_id':request['request_id'],'seed':request['seed'],
                          'prompt_ids_sha256':request['prompt_ids_sha256'],'rank':rank}:
                raise ValueError("request capsule identity differs")
            if (row['index']!=index or row['rank']!=rank or row['request_id']!=request['request_id']
                    or row['complete'] is not True or row['generated_tokens']!=len(ids)
                    or row['token_ids_sha256']!=digest or row['finish_reason']!=events[-1]['finish_reason']
                    or row['protocol_sha256']!=sha256(protocol.canonical(plan)).hexdigest()
                    or row['first_token_delivered_before_decode'] is not True
                    or row['live_session_resume'] is not (len(ids)>1)
                    or len(row['decode_seconds'])!=len(ids)-1
                    or any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in row['decode_seconds'])):
                raise ValueError("native fleet tokens/policy/phase timings differ")
            if rank==0:
                if (row['delivery_boundary']!='rank0_local_jsonl_write_flush'
                        or any(type(row[n]) not in (int,float) or not math.isfinite(row[n]) or row[n]<0
                               for n in ('ttft_seconds','delivered_request_seconds'))
                        or row['delivered_request_seconds']<row['ttft_seconds']):
                    raise ValueError("native first-delivery timing differs")
            elif row['ttft_seconds'] is not None or row['delivered_request_seconds'] is not None:
                raise ValueError("nonoutput rank cannot claim delivery timing")
            checked=replay_observation(directory,row,prompt_tokens=len(request['prompt_ids']),parent=parent,
                full_index_layers=full_index_layers)
            maximum_peak=max(maximum_peak,checked['peak'])
            if checked['trace'] is not None:
                trace=checked['trace']
                path=root/trace['path']
                if (not path.is_relative_to(root/f'native_trace.rank{rank}') or trace['request_index']!=index
                        or trace['graph']!='observer' or trace['actual_model_calls']!=1
                        or row['observations']['traced_decode_indices']!=[0]):
                    raise ValueError("native trace original scope differs")
                data=read(path,TRACE_CAP)
                if len(data)!=trace['bytes'] or sha256(data).hexdigest()!=trace['sha256']:
                    raise ValueError("native trace bytes differ")
                traces.append((rank,path))
        score=score_answer(request,ids,tokenizer,registry)
        text=score.pop('text').encode()
        if read(directories[0]/'answer.txt.gz',16<<20,compressed=True)!=text:
            raise ValueError("answer text does not decode from original delivered tokens")
        if any(rows[0].get(k)!=v for k,v in score.items()):raise ValueError("native score replay differs")
        results.append(dict(index=index,request_id=request['request_id'],dataset=request['dataset'],
            generated_tokens=len(ids),token_ids_sha256=digest,**score,ttft_seconds=rows[0]['ttft_seconds']))
    benchmark={}
    for name,count in protocol.COUNTS.items():
        subset=[r for r in results if r['dataset']==name]
        scored=[r for r in subset if r['correct'] is not None]
        full=len(scored)==count
        correct=sum(r['correct'] for r in scored)
        interval=protocol.wilson(correct,count) if full else None
        target=plan['targets'][name]
        benchmark[name]=dict(registered=count,completed=len(subset),scored=len(scored),correct=correct,
            score=correct/count if full else None,target=target,
            score_gap=correct/count-target if full else None,wilson95=interval,
            material_deficit=interval[1]<target-0.05 if full else None,
            status='FULL_SET_SCORED_PROTOCOL_CAVEATS_APPLY' if full else 'INCOMPLETE_OR_UNSCORED')
    return dict(schema='ws32_native_request_replay_v1',completed_requests=len(results),items=results,
        partial=partial,benchmarks=benchmark,maximum_peak_hbm_bytes=maximum_peak,
        trace_originals=[dict(rank=r,path=str(p.relative_to(root))) for r,p in traces],
        trace_physical_coverage_verified=False,protected_result_sealed=False,
        matched_card_parity_claim=False,delivery_boundary='rank0 local JSONL write/flush, not network service')
