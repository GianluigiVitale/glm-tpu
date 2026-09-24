"""The file-inbox client of a resident controller, used by the chat UI and the OpenAI-compatible API."""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from glm_tpu.entrypoints.openai import protocol
from glm_tpu.entrypoints.openai.tool_parser import final_channel
from glm_tpu.utils.io_utils import atomic


class Resident:
    def __init__(self, run, dispatch, repo, model_path):
        from glm_tpu.config import model
        from glm_tpu.engine import request
        from transformers import AutoTokenizer
        self.run = run
        self.identity = json.loads(dispatch.read_text())
        self.check()
        initial = request.read(Path(self.identity['command'][self.identity['command'].index('--request') + 1]))
        self.capacity = initial['context_capacity']
        if self.capacity not in request.CAPACITIES:
            raise ValueError('The resident session uses an unsupported context capacity.')
        if json.loads((run / 'resident-measurement.json').read_text())['code_hash'] != self.identity['code_hash']:
            raise ValueError('Resident run does not match controller provenance.')
        self.lease = (run / 'benchmark-producer.lock').open('a')
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.template = model.verified_template(repo, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)

    def check(self):
        proc = Path('/proc') / str(self.identity['pid'])
        try:
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            valid = (stat[19] == str(self.identity['start_ticks']) and stat[0] not in ('Z', 'X')
                     and b'glm_tpu.executor.multihost_executor' in argv
                     and b'--keep-loaded' in argv and not (self.run / 'inbox/stop.json').exists())
        except (OSError, IndexError):
            valid = False
        if not valid:
            raise ValueError('The resident model is unavailable. No model was restarted.')

    def prepare(self, messages, key):
        from glm_tpu.engine import request
        self.check()
        return request.from_messages(messages, tokenizer=self.tokenizer, chat_template=self.template,
                                     request_id='ui-' + key, context_capacity=self.capacity)

    def prepare_api(self, messages, key, *, tools, effort, max_new_tokens, context_capacity):
        """Render a complete stateless chat, including tools, at the pinned profile."""
        from glm_tpu.engine import request
        from glm_tpu.engine.request import MAX_NEW
        self.check()
        ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=False,
            chat_template=self.template, enable_thinking=True, reasoning_effort=effort)
        remaining = context_capacity - len(ids)
        if remaining <= 0:
            raise protocol.ApiError('this conversation fills the %d-slot context; send less history '
                               'or smaller tool output' % context_capacity)
        budget = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        budget = min(budget, MAX_NEW)
        return request.from_token_ids(ids, request_id='api-' + key, max_new_tokens=budget,
                                      context_capacity=context_capacity)

    def next_sequence(self):
        self.check()
        ready = json.loads((self.run / 'resident-ready.json').read_text())['sequence']
        job = self.run if ready == 0 else self.run / f'resident-{ready:04d}'
        if not (job / 'resident-measurement.json').exists():
            raise ValueError('The resident controller is still collecting its previous answer.')
        if any(p.stem.isdigit() and int(p.stem) > ready for p in (self.run / 'inbox').glob('*.json')):
            raise ValueError('The resident inbox has another request owner.')
        return ready + 1

    def publish(self, job, state):
        from glm_tpu.engine import request
        self.check()
        request.validate(job['payload'])
        source = state / (job['id'] + '.request.json')
        if not source.exists():
            atomic(source, job['payload'])
        elif json.loads(source.read_text()) != job['payload']:
            raise ValueError('Saved request identity differs; preserved for inspection.')
        target = self.run / 'inbox' / f"{job['sequence']:04d}.json"
        try:
            os.link(source, target)  # Publish once, never overwrite another producer.
        except FileExistsError:
            if target.is_symlink() or json.loads(target.read_text()) != job['payload']:
                raise ValueError('Resident sequence belongs to a different request.')

    def observe(self, job):
        root = self.run / f"resident-{job['sequence']:04d}"
        token_file = root / 'tokens.jsonl'
        events = []
        if token_file.exists():
            for line in token_file.read_bytes().splitlines(keepends=True):
                if line.endswith(b'\n'):
                    events.append(json.loads(line))
        if any(x['request_id'] != job['payload']['request_id'] or x['index'] != i for i, x in enumerate(events)):
            raise ValueError('Token stream identity differs.')
        text = self.tokenizer.decode([x['token_id'] for x in events], skip_special_tokens=False)
        thinking, answer = final_channel(text)
        update = dict(thinking=thinking, answer=answer, output_tokens=len(events), status='generating')
        receipt = root / 'resident-measurement.json'
        if not receipt.exists():
            self.check()
            return update
        summary = json.loads(receipt.read_text())
        rows = [json.loads((root / f'runner.rank{i}.json').read_text()) for i in range(8)]
        reports = [row['requests'][0] for row in rows]
        digest = hashlib.sha256(b''.join(x['token_id'].to_bytes(4, 'little', signed=True) for x in events)).hexdigest()
        if not (summary['passed'] and summary['all_ranks_agree'] and summary['model_retained']
                and summary['resident_sequence'] == job['sequence']
                and summary['code_hash'] == self.identity['code_hash']):
            raise ValueError('Resident completion check failed.')
        for rank, (row, report) in enumerate(zip(rows, reports)):
            if not (row['complete'] and row['rank'] == rank and row['code_hash'] == self.identity['code_hash']
                    and row['request_sha256'] == job['payload']['request_sha256']
                    and report['request_sha256'] == job['payload']['request_sha256']
                    and report['token_sha256'] == digest and report['emitted'] == len(events)):
                raise ValueError('All-host request/token agreement failed.')
            if report['output_budget_tokens'] != job['payload']['max_new_tokens']:
                raise ValueError('Output allowance differs from the prepared request.')
            for name, program in row['programs'].items():
                if not program['memory_admission']['passed'] or ('hlo_admission' in program and not program['hlo_admission']['passed']):
                    raise ValueError('Graph/memory admission failed.')
                if any(program[k] != rows[0]['programs'][name][k] for k in ('stablehlo_sha256', 'optimized_hlo_sha256')):
                    raise ValueError('All-host graph identity differs.')
        causes = {r['stop_cause'] for r in reports}
        if len(causes) != 1:
            raise ValueError('All-host stopping conditions differ.')
        thinking, answer = final_channel((root / 'answer.txt').read_text())
        complete = causes == {'eos'} and all(r['finish_reason'] == 'eos' for r in reports) and bool(answer)
        update.update(thinking=thinking, answer=answer, status='complete' if complete else 'incomplete',
                      stop_cause=reports[0]['stop_cause'],
                      decode_tps=min(r['decode_tokens_per_second'] or 0 for r in reports),
                      prefill_seconds=max(r['prefill_seconds'] for r in reports))
        return update
