"""Simple private question submission using the existing protected controller."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from glm_tpu import user_request as legacy
from . import request


def prepare_questions(questions, *, repo, tokenizer_root, output_root,
                      context_capacity, max_new_tokens, concurrent=False):
    """Prepare once; all questions are queued before the single fleet launch."""
    if (type(questions) is not list or not 1 <= len(questions) <= 10
            or any(type(q) is not str or not q.strip() for q in questions)):
        raise ValueError('provide one to ten nonempty question strings')
    if output_root.resolve().is_relative_to(repo.resolve()):
        raise ValueError('private questions must stay outside the repository')
    if concurrent and (len(questions)>8 or context_capacity!=request.CONCURRENT_CAPACITY):
        raise ValueError('concurrent mode accepts up to eight questions with --context 32k')
    if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 1):
        raise ValueError('output budget must be a positive integer')
    output_root.mkdir(mode=0o700)
    values=[]
    for index,question in enumerate(questions):
        messages=output_root/f'messages-{index:03d}.json'
        fd=os.open(messages,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(legacy.canonical([dict(role='user',content=question)])+b'\n')
        prepared=output_root/f'prepared-{index:03d}.json'
        request.prepare_file(messages_path=messages,output=prepared,repo=repo,
            tokenizer_root=tokenizer_root,request_id=f'question-{index+1:02d}',
            max_new_tokens=max_new_tokens,context_capacity=context_capacity)
        values.append(request.read(prepared))
    value=request.batch(values,concurrent=concurrent)
    raw=legacy.canonical(value)+b'\n'
    if len(raw)>legacy.PAYLOAD_CAP:raise ValueError('combined request payload is too large')
    path=output_root/'request.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return path


def main(args):
    # No JAX/device import occurs until the protected worker is dispatched.
    from scripts.release import launch_ws32_optimized_request as launch
    if args.questions is not None:
        questions=json.loads(legacy.read_bounded(args.questions,10*legacy.MESSAGES_CAP))
    else:questions=[args.question]
    capacity={'8k':request.CAPACITY,'32k':request.CONCURRENT_CAPACITY,
              '128k':request.LONG_CAPACITY}[args.context]
    budget=args.max_new_tokens
    if budget is None and args.context in ('8k','32k'):budget=2048
    root=launch.worker.RUN_ROOT/('ordinary_inputs_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    path=prepare_questions(questions,repo=launch.REPO,tokenizer_root=launch.worker.TOKENIZER,
        output_root=root,context_capacity=capacity,max_new_tokens=budget,
        concurrent=getattr(args,'concurrent',False))
    print('PRIVATE_INPUT '+str(path),flush=True)
    if args.prepare_only:return 0
    return launch.main(['--request',str(path),'--wall-seconds',str(args.wall_seconds),'--print-answers'])
