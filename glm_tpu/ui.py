"""Loopback chat UI attached to an existing resident GLM controller.

This process never initializes devices, starts workers, or sends a stop command.
Private chat history and requests live outside the checkout.
"""
import argparse
import fcntl
import secrets
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import uuid

from . import api


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        os.chmod(tmp, 0o600)
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def final_channel(text):
    if '</think>' not in text:
        return text.removeprefix('<think>').strip(), ''
    thinking, answer = text.split('</think>', 1)
    answer = re.split(r'<\|(?:user|endoftext|observation)\|>', answer, maxsplit=1)[0]
    return thinking.removeprefix('<think>').strip(), answer.strip()


class Resident:
    def __init__(self, run, dispatch, repo):
        from .optimized import model, request
        from transformers import AutoTokenizer
        self.run = run
        self.identity = json.loads(dispatch.read_text())
        self.check()
        initial = request.read(Path(self.identity['command'][self.identity['command'].index('--request') + 1]))
        if initial['context_capacity'] != 32768:
            raise ValueError('The UI requires the resident 32K profile.')
        if json.loads((run / 'resident-measurement.json').read_text())['code_hash'] != self.identity['code_hash']:
            raise ValueError('Resident run does not match controller provenance.')
        self.lease = (run / 'benchmark-producer.lock').open('a')
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.template = model.verified_template(repo, model.TOKENIZER_ROOT)
        self.tokenizer = AutoTokenizer.from_pretrained(model.TOKENIZER_ROOT, local_files_only=True, trust_remote_code=False)

    def check(self):
        proc = Path('/proc') / str(self.identity['pid'])
        try:
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            valid = (stat[19] == str(self.identity['start_ticks']) and stat[0] not in ('Z', 'X')
                     and b'scripts.release.launch_ws32_optimized_request' in argv
                     and b'--keep-loaded' in argv and not (self.run / 'inbox/stop.json').exists())
        except (OSError, IndexError):
            valid = False
        if not valid:
            raise ValueError('The resident model is unavailable. No model was restarted.')

    def prepare(self, messages, key):
        from .optimized import request
        self.check()
        return request.from_messages(messages, tokenizer=self.tokenizer, chat_template=self.template,
                                     request_id='ui-' + key, context_capacity=32768)

    def prepare_api(self, messages, key, *, tools, effort, max_new_tokens, context_capacity):
        """Render a complete stateless chat, including tools, at the pinned profile."""
        from .optimized import request
        self.check()
        ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=False,
            chat_template=self.template, enable_thinking=True, reasoning_effort=effort)
        remaining = context_capacity - len(ids)
        if remaining <= 0:
            raise api.ApiError('this conversation fills the %d-slot context; send less history '
                               'or smaller tool output' % context_capacity)
        if max_new_tokens is None:
            budget = remaining
        else:
            if type(max_new_tokens) is not int or max_new_tokens <= 0:
                raise api.ApiError('max_tokens must be a positive integer')
            budget = min(max_new_tokens, remaining)
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
        from .optimized import request
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


class Chats:
    def __init__(self, root, backend):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_mode & 0o077:
            raise ValueError('Chat directory must be private (mode 0700).')
        self.root, self.backend = root, backend
        self.mutex = threading.RLock()
        self.db = json.loads((root / 'chats.json').read_text()) if (root / 'chats.json').exists() else dict(chats=[], jobs=[])
        self.error = None
        self.stopping = threading.Event()

    def save(self):
        atomic(self.root / 'chats.json', self.db)

    def snapshot(self):
        with self.mutex:
            availability = self.error
            try:
                self.backend.check()
            except (ValueError, OSError) as exc:
                availability = str(exc)
            # Exclude private model payloads from browser responses.
            jobs = [{k: v for k, v in j.items() if k != 'payload'}
                    for j in self.db['jobs'] if not j.get('api')]
            return json.loads(json.dumps(dict(chats=self.db['chats'], jobs=jobs, error=availability,
                                             model='GLM-5.3', capacity=32768, scheduling='sequential')))

    def change(self, data):
        with self.mutex:
            action = data.get('action')
            if action == 'create':
                chat = dict(id=uuid.uuid4().hex, title='New conversation', messages=[])
                self.db['chats'].append(chat)
                self.save()
                return dict(id=chat['id'])
            chat = next((c for c in self.db['chats'] if c['id'] == data.get('chat')), None)
            if chat is None:
                raise ValueError('Conversation not found.')
            if action == 'rename':
                title = data.get('title')
                if type(title) is not str or not 1 <= len(title.strip()) <= 100:
                    raise ValueError('Use a title between 1 and 100 characters.')
                chat['title'] = title.strip()
            elif action == 'send':
                key = data.get('id')
                text = data.get('text')
                if type(key) is not str or not re.fullmatch('[a-f0-9]{32}', key):
                    raise ValueError('Invalid message identity.')
                existing = next((j for j in self.db['jobs'] if j['id'] == key), None)
                if existing:
                    if existing['chat'] != chat['id'] or existing['user_text'] != text:
                        raise ValueError('Message identity was already used.')
                    return dict(id=key)
                if type(text) is not str or not text.strip() or len(text.encode()) > 128000:
                    raise ValueError('Enter a message of at most 128,000 bytes.')
                if self.error:
                    raise ValueError(self.error)
                if len([j for j in self.db['jobs'] if j['status'] in ('queued', 'generating')]) >= PENDING:
                    raise ValueError('Ten messages are already waiting. Wait for an answer.')
                if chat['messages'] and chat['messages'][-1].get('status') != 'complete':
                    raise ValueError('Wait for a completed answer, or start a new conversation.')
                messages = [dict(role=m['role'], content=m['content']) for m in chat['messages']]
                messages.append(dict(role='user', content=text))
                payload = self.backend.prepare(messages, key)
                job = dict(id=key, chat=chat['id'], user_text=text, status='queued', payload=payload,
                           sequence=None, answer='', thinking='', output_tokens=0,
                           prompt_tokens=len(payload['prompt_ids']), created=time.time())
                self.db['jobs'].append(job)
                chat['messages'].extend([dict(role='user', content=text),
                                         dict(role='assistant', content='', status='queued', job=key)])
                if len(chat['messages']) == 2 and chat['title'] == 'New conversation':
                    chat['title'] = ' '.join(text.split())[:55]
            elif action == 'delete':
                if any(j['chat'] == chat['id'] and j['status'] in ('queued', 'generating') for j in self.db['jobs']):
                    raise ValueError('Wait for this conversation to finish before deleting it.')
                self.db['chats'].remove(chat)
                # Original inference evidence stays in the private resident run.
                self.db['jobs'] = [j for j in self.db['jobs'] if j['chat'] != chat['id']]
            else:
                raise ValueError('Unknown action.')
            self.save()
            return dict(ok=True)

    def submit(self, payload, key, *, label):
        """Queue a stateless request; it belongs to no saved conversation."""
        with self.mutex:
            existing = next((j for j in self.db['jobs'] if j['id'] == key), None)
            if existing is not None:
                return existing
            if self.error:
                raise api.ApiError(self.error, status=503, kind='server_error')
            if len([j for j in self.db['jobs'] if j['status'] in ('queued', 'generating')]) >= PENDING:
                raise api.ApiError('too many requests are already waiting for this one model',
                                   status=429, kind='rate_limit_error')
            job = dict(id=key, chat=None, api=True, label=label, user_text='', status='queued',
                       payload=payload, sequence=None, answer='', thinking='', output_tokens=0,
                       prompt_tokens=len(payload['prompt_ids']), created=time.time())
            self.db['jobs'].append(job)
            self.retire()
            self.save()
            return job

    def job(self, key):
        with self.mutex:
            found = next((j for j in self.db['jobs'] if j['id'] == key), None)
            return None if found is None else json.loads(json.dumps(
                {k: v for k, v in found.items() if k != 'payload'}))

    def retire(self):
        """Bound the stateless history; saved conversations are never touched."""
        finished = [j for j in self.db['jobs']
                    if j.get('api') and j['status'] in ('complete', 'incomplete')]
        for job in finished[:-API_HISTORY] if len(finished) > API_HISTORY else []:
            self.db['jobs'].remove(job)

    def step(self):
        with self.mutex:
            job = next((j for j in self.db['jobs'] if j['status'] in ('queued', 'generating')), None)
            if job is None or self.error:
                return
            try:
                if job['sequence'] is None:
                    job['sequence'] = self.backend.next_sequence()
                    self.save()  # Durable identity before any inbox write.
                self.backend.publish(job, self.root)
                update = self.backend.observe(job)
                job.update(update)
                if job.get('chat'):
                    chat = next(c for c in self.db['chats'] if c['id'] == job['chat'])
                    message = next(m for m in chat['messages'] if m.get('job') == job['id'])
                    message.update(content=job['answer'], status=job['status'])
                self.save()
            except Exception as exc:
                # Preserve admission identity. Restart can reconcile the SAME request;
                # never resubmit under a fresh sequence after an uncertain failure.
                self.error = str(exc)
                job['error'] = self.error
                atomic(self.root / 'error.json', dict(error=self.error, job=job['id'], time=time.time()))

    def work(self):
        while not self.stopping.is_set():
            self.step()
            with self.mutex:
                active = any(j['status'] == 'generating' for j in self.db['jobs'])
            self.stopping.wait(0.25 if active else 1)


PENDING = 10
API_HISTORY = 50


def api_token(path):
    """One local key, created 0600 on first use and never written to a log."""
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError('The API key file must be private (mode 0600).')
        token = path.read_text().strip()
        if not token:
            raise ValueError('The API key file is empty.')
        return token
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(token + '\n')
    return token


def handler(chats, service=None, token=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not put private questions in access logs.

        def respond(self, code, value, kind='application/json'):
            raw = json.dumps(value).encode() if kind == 'application/json' else value
            self.send_response(code)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(raw)

        def bearer(self):
            # The browser UI is same-origin; API clients authenticate with the local key.
            header = self.headers.get('Authorization', '')
            offered = header[7:] if header.startswith('Bearer ') else ''
            return token is not None and secrets.compare_digest(offered, token)

        def stream(self, source):
            try:
                first = next(source)
            except api.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except StopIteration:
                return self.respond(500, dict(error=dict(message='empty stream')))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(first)
                self.wfile.flush()
                for chunk in source:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except api.ApiError as exc:
                # Headers are already sent; report in-band and end the stream.
                self.wfile.write(b'data: ' + json.dumps(exc.body()).encode() + b'\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client left; the admitted request still finishes.

        def allowed(self):
            # Loopback binding plus Host/Origin checks prevent DNS rebinding and CSRF.
            host = self.headers.get('Host', '')
            expected = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return host in expected and (origin is None or origin == 'http://' + host)

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, dict(error='Local origin required.'))
            path = urlsplit(self.path).path
            if path.startswith('/v1/'):
                if not self.bearer():
                    return self.respond(401, dict(error=dict(message='A local API key is required.',
                                                             type='invalid_request_error')))
                if path != '/v1/models':
                    return self.respond(404, dict(error=dict(message='Not found.',
                                                             type='invalid_request_error')))
                return self.respond(200, service.models())
            if path == '/api/state':
                return self.respond(200, chats.snapshot())
            assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                      '/style.css': ('style.css', 'text/css')}
            if path not in assets:
                return self.respond(404, dict(error='Not found.'))
            name, kind = assets[path]
            self.respond(200, (Path(__file__).parent / 'web' / name).read_bytes(), kind)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path.startswith('/v1/'):
                return self.completions(path)
            if not self.allowed() or self.headers.get('X-GLM-UI') != '1':
                return self.respond(403, dict(error='Local UI required.'))
            if self.path != '/api/chat':
                return self.respond(404, dict(error='Not found.'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 256000:
                    raise ValueError('Message is too large.')
                self.connection.settimeout(10)
                data = json.loads(self.rfile.read(size))
                if type(data) is not dict:
                    raise ValueError('Expected an object.')
                result = chats.change(data)
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=str(exc)))
            self.respond(200, result)

        def completions(self, path):
            if not self.allowed():
                return self.respond(403, dict(error=dict(message='Local origin required.',
                                                         type='invalid_request_error')))
            if not self.bearer():
                return self.respond(401, dict(error=dict(message='A local API key is required.',
                                                         type='invalid_request_error')))
            if path != '/v1/chat/completions':
                return self.respond(404, dict(error=dict(message='Not found.',
                                                         type='invalid_request_error')))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4 << 20:
                    raise api.ApiError('request body is empty or too large')
                self.connection.settimeout(30)
                data = json.loads(self.rfile.read(size))
            except api.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=dict(message=str(exc),
                                                         type='invalid_request_error')))
            self.connection.settimeout(None)
            if type(data) is dict and data.get('stream'):
                return self.stream(service.stream(data))
            try:
                return self.respond(200, service.completion(data))
            except api.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=dict(message=str(exc),
                                                         type='invalid_request_error')))
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dispatch', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--api-key-file', type=Path,
                        help='default: <state>/api-key, created 0600 on first start')
    parser.add_argument('--no-api', action='store_true',
                        help='serve only the browser workspace, without /v1')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.state.resolve().is_relative_to(args.repo.resolve()):
        parser.error('Private chat state must be outside the source checkout.')
    backend = Resident(args.run.resolve(), args.dispatch.resolve(), args.repo.resolve())
    chats = Chats(args.state, backend)
    service = token = key_path = None
    if not args.no_api:
        key_path = args.api_key_file or (args.state / 'api-key')
        token = api_token(key_path)
        service = api.Api(chats, backend)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(chats, service, token))
    thread = threading.Thread(target=chats.work, daemon=True)
    thread.start()
    print(f'GLM-5.3 chat: http://127.0.0.1:{args.port} (existing model retained)', flush=True)
    if service is not None:
        print(f'OpenAI-compatible API: http://127.0.0.1:{args.port}/v1 '
              f'(key in {key_path})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        chats.stopping.set()
        thread.join(timeout=5)
        server.server_close()
        # Closing this UI does not stop the resident controller or its workers.


if __name__ == '__main__':
    main()
