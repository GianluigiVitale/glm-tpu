"""CPU-only tests of UI boundaries; no tokenizer, checkpoint, cloud or TPU."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_tpu.entrypoints.serve.job_queue import JobQueue
from glm_tpu.engine.resident_client import Resident
from glm_tpu.entrypoints.serve.server import ThreadingHTTPServer
from glm_tpu.entrypoints.openai.tool_parser import final_channel
from glm_tpu.entrypoints.serve.http_handler import handler
from glm_tpu.entrypoints.ui.conversations import ConversationStore


class FakeResident:
    def __init__(self):
        self.messages = []
        self.published = {}
        self.sequence = 770
        self.complete = False

    def check(self):
        pass

    def prepare(self, messages, key):
        if len(messages[-1]["content"]) > 100:
            raise ValueError("context capacity exceeded")
        self.messages.append(messages)
        return dict(prompt_ids=[1, 2], max_new_tokens=32766, request_id=key)

    def next_sequence(self):
        return self.sequence + 1

    def publish(self, job, root):
        old = self.published.setdefault(job["sequence"], job["payload"])
        assert old == job["payload"]

    def observe(self, job):
        self.sequence = job["sequence"]
        return dict(
            status="complete" if self.complete else "generating",
            answer="42" if self.complete else "",
            thinking="working",
            output_tokens=3,
            decode_tps=13.5,
            stop_cause="eos",
        )


def open_store(root, backend):
    """The browser workspace as the server composes it: a conversation store over the job queue."""
    return ConversationStore(JobQueue(root, backend))


def chat(store):
    return store.change(dict(action="create"))["id"]


def send(store, c, text="question", key="a" * 32):
    return store.change(dict(action="send", chat=c, text=text, id=key))


def test_idempotence_and_restart_preserve_admitted_identity(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    c = chat(store)
    send(store, c)
    send(store, c)
    assert len(store.queue.db["jobs"]) == 1
    with pytest.raises(ValueError, match="already used"):
        send(store, c, "different")
    store.queue.step()
    restarted = open_store(tmp_path, backend)
    backend.complete = True
    restarted.queue.step()
    assert len(backend.published) == 1
    assert restarted.snapshot()["jobs"][0]["status"] == "complete"
    assert restarted.snapshot()["jobs"][0]["sequence"] == 771


def test_crash_after_publish_reconciles_same_sequence(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    c = chat(store)
    send(store, c)
    original = backend.observe
    backend.observe = lambda _: (_ for _ in ()).throw(OSError("receipt unreadable"))
    store.queue.step()
    assert store.queue.error and len(backend.published) == 1
    assert json.loads((tmp_path / "chats.json").read_text())["jobs"][0]["sequence"] == 771
    backend.observe = original
    backend.complete = True
    restarted = open_store(tmp_path, backend)
    restarted.queue.step()
    assert len(backend.published) == 1


def test_queue_runs_one_at_a_time(tmp_path):
    backend = FakeResident()
    store = open_store(tmp_path, backend)
    a, b = chat(store), chat(store)
    send(store, a)
    send(store, b, key="b" * 32)
    store.queue.step()
    store.queue.step()
    assert list(backend.published) == [771]
    backend.complete = True
    store.queue.step()
    store.queue.step()
    assert list(backend.published) == [771, 772]


def test_reasoning_is_not_final_answer():
    assert final_channel("The answer might be 42") == ("The answer might be 42", "")
    assert final_channel("<think>work</think>42<|user|>") == ("work", "42")
    assert final_channel("work</think>42<|endoftext|>") == ("work", "42")
    assert final_channel("work</think>42<|observation|>") == ("work", "42")


def test_http_csrf_rebinding_static_and_private_payload(tmp_path):
    store = open_store(tmp_path, FakeResident())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert b"GLM-5.3" in response.read()
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        for headers in (
            {},
            {"X-GLM-UI": "1", "Origin": "https://evil.example"},
            {"X-GLM-UI": "1", "Host": "evil.example"},
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/api/chat", data=b'{"action":"create"}', headers=headers))
            assert error.value.code == 403
        with urlopen(Request(base + "/api/chat", data=b'{"action":"create"}', headers={"X-GLM-UI": "1"})) as response:
            assert json.loads(response.read())["id"]
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../ui.py")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_atomic_publication_never_overwrites_a_different_request(tmp_path):
    from glm_tpu.engine.request import from_token_ids

    backend = Resident.__new__(Resident)
    backend.run = tmp_path / "run"
    (backend.run / "inbox").mkdir(parents=True)
    backend.check = lambda: None
    state = tmp_path / "state"
    state.mkdir()
    payload = from_token_ids([1, 2], request_id="ui-test", max_new_tokens=32766, context_capacity=32768)
    job = dict(id="test", sequence=771, payload=payload)
    backend.publish(job, state)
    backend.publish(job, state)
    target = backend.run / "inbox/0771.json"
    assert json.loads(target.read_text()) == payload
    assert target.stat().st_mode & 0o077 == 0
    target.unlink()
    target.write_text('{"different":true}')
    with pytest.raises(ValueError, match="different request"):
        backend.publish(job, state)
    assert target.read_text() == '{"different":true}'


@pytest.mark.parametrize("corruption", [None, "tokens", "budget", "memory", "graphs"])
def test_completion_requires_matching_eight_host_receipts(tmp_path, corruption):
    import hashlib

    backend = Resident.__new__(Resident)
    backend.run = tmp_path
    backend.identity = dict(code_hash="code")
    backend.check = lambda: None
    backend.tokenizer = type("Tokenizer", (), {"decode": lambda *args, **kwargs: "thinking</think>42<|user|>"})()
    root = tmp_path / "resident-0771"
    root.mkdir()
    event = dict(token_id=42, request_id="ui-test", index=0)
    (root / "tokens.jsonl").write_text(json.dumps(event) + "\n")
    (root / "answer.txt").write_text("thinking</think>42<|user|>")
    summary = dict(passed=True, all_ranks_agree=True, model_retained=True, resident_sequence=771, code_hash="code")
    (root / "resident-measurement.json").write_text(json.dumps(summary))
    digest = hashlib.sha256((42).to_bytes(4, "little", signed=True)).hexdigest()
    for rank in range(8):
        report = dict(
            request_sha256="request",
            token_sha256=digest,
            emitted=1,
            output_budget_tokens=32766,
            stop_cause="eos",
            finish_reason="eos",
            decode_tokens_per_second=13.5,
            prefill_seconds=1,
        )
        program = dict(memory_admission=dict(passed=True), stablehlo_sha256="hlo", optimized_hlo_sha256="hlo")
        if rank == 7:
            if corruption == "tokens":
                report["token_sha256"] = "different"
            if corruption == "budget":
                report["output_budget_tokens"] = 1024
            if corruption == "memory":
                program["memory_admission"]["passed"] = False
            if corruption == "graphs":
                program["stablehlo_sha256"] = "different"
        row = dict(
            complete=True,
            rank=rank,
            code_hash="code",
            request_sha256="request",
            requests=[report],
            programs=dict(decode=program),
        )
        (root / f"runner.rank{rank}.json").write_text(json.dumps(row))
    job = dict(sequence=771, payload=dict(request_id="ui-test", request_sha256="request", max_new_tokens=32766))
    if corruption:
        with pytest.raises(ValueError):
            backend.observe(job)
    else:
        result = backend.observe(job)
        assert result["status"] == "complete" and result["answer"] == "42"
