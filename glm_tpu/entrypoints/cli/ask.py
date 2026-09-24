"""Simple private question submission using the existing protected controller."""
from datetime import datetime, timezone
import json
import os

from glm_tpu.utils import json_utils
from glm_tpu.utils import io_utils
from glm_tpu.engine import request


def prepare_questions(questions, *, repo, tokenizer_root, output_root,
                      context_capacity, max_new_tokens, concurrent=False):
    """Prepare once; all questions are queued before the single fleet launch."""
    if (type(questions) is not list or not 1 <= len(questions) <= 10
            or any(type(q) is not str or not q.strip() for q in questions)):
        raise ValueError('provide one to ten nonempty question strings')
    if output_root.resolve().is_relative_to(repo.resolve()):
        raise ValueError('private questions must stay outside the repository')
    if concurrent and (len(questions)>request.CONCURRENT_LIMIT or context_capacity!=request.CONCURRENT_CAPACITY):
        raise ValueError('concurrent mode accepts up to four questions with --context 32k')
    if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 1):
        raise ValueError('output budget must be a positive integer')
    output_root.mkdir(mode=0o700)
    values=[]
    for index,question in enumerate(questions):
        messages=output_root/f'messages-{index:03d}.json'
        fd=os.open(messages,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:
            stream.write(json_utils.canonical([dict(role='user',content=question)])+b'\n')
        prepared=output_root/f'prepared-{index:03d}.json'
        request.prepare_file(messages_path=messages,output=prepared,repo=repo,
            tokenizer_root=tokenizer_root,request_id=f'question-{index+1:02d}',
            max_new_tokens=max_new_tokens,context_capacity=context_capacity)
        values.append(request.read(prepared))
    value=request.batch(values,concurrent=concurrent)
    raw=json_utils.canonical(value)+b'\n'
    if len(raw)>request.PAYLOAD_CAP:raise ValueError('combined request payload is too large')
    path=output_root/'request.json'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return path


def main(args):
    # No JAX/device import occurs until the protected worker is dispatched.
    from glm_tpu.config.site import SiteConfig
    from glm_tpu.executor import multihost_executor as launch
    site=SiteConfig.load(getattr(args,'site',None))
    if args.questions is not None:
        questions=json.loads(io_utils.read_bounded(args.questions,10*request.MESSAGES_CAP))
    else:questions=[args.question]
    capacity={'8k':request.CAPACITY,'32k':request.CONCURRENT_CAPACITY,
              '128k':request.LONG_CAPACITY,'256k':request.AGENT_CAPACITY}[args.context]
    budget=args.max_new_tokens
    root=site.paths.run_root/('ordinary_inputs_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    path=prepare_questions(questions,repo=launch.REPO,tokenizer_root=site.paths.model_path,
        output_root=root,context_capacity=capacity,max_new_tokens=budget,
        concurrent=getattr(args,'concurrent',False))
    print('PRIVATE_INPUT '+str(path),flush=True)
    if args.prepare_only:return 0
    return launch.main(['--request',str(path),'--wall-seconds',str(args.wall_seconds),'--print-answers']+
                       (['--keep-loaded'] if getattr(args,'keep_loaded',False) else [])+
                       (['--site',str(args.site)] if getattr(args,'site',None) else []))
