"""CPU-only tests of UI boundaries; no tokenizer, checkpoint, cloud or TPU."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from glm_tpu.entrypoints.serve.job_queue import JobQueue
from glm_tpu.engine.resident_client import Resident
from glm_tpu.entrypoints.openai.serving_chat import OpenAIServingChat
from glm_tpu.entrypoints.openai.serving_models import OpenAIServingModels
from glm_tpu.entrypoints.serve.server import ThreadingHTTPServer
from glm_tpu.entrypoints.openai.tool_parser import final_channel
from glm_tpu.entrypoints.serve.http_handler import handler
from glm_tpu.entrypoints.ui.conversations import ConversationStore
from tests.fixtures.serving import FakeResident, open_store


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


def test_served_page_shows_the_session_capacity_not_a_fixed_32k(tmp_path):
    # H17: the footer starts neutral and render() writes "<capacity/1024>K context" from /api/state.
    store = open_store(tmp_path, FakeResident(capacity=166912))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            page = response.read().decode()
        with urlopen(base + "/app.js") as response:
            script = response.read().decode()
        with urlopen(base + "/api/state") as response:
            state = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "32K" not in page and "32K" not in script
    assert '<p class="footer-note" id="contextNote">' in page
    assert "$('contextNote').textContent=`${Math.round(state.capacity/1024)}K context," in script
    assert state["capacity"] == 166912


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


class StubTokenizer:
    """Renders any chat to ``length`` token ids: the context-overflow refusal needs no real tokenizer."""

    def __init__(self, length):
        self.length = length

    def apply_chat_template(self, messages, **options):
        return [1] * self.length


def resident(prompt_tokens):
    """A ``Resident`` without its controller: its real ``prepare_api`` over a stub tokenizer."""
    backend = Resident.__new__(Resident)
    backend.check = lambda: None
    backend.template = None
    backend.tokenizer = StubTokenizer(prompt_tokens)
    return backend


def completions(tmp_path, backend, body):
    """POST ``body`` to ``/v1/chat/completions`` of the server as ``serve.server`` composes it (the queue with its
    conversation store, the chat and model services at 32K) while the queue worker runs; return the status, the
    content type, the body bytes and the queue."""
    queue = JobQueue(tmp_path / "state", backend)
    chat = OpenAIServingChat(queue, backend, capacity=32768)
    models = OpenAIServingModels(capacity=chat.capacity, wait_seconds=chat.wait_seconds)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(ConversationStore(queue), chat, models, "secret-key"))
    stop = threading.Event()

    def work():
        while not stop.is_set():
            queue.step()
            stop.wait(0.01)

    threads = [threading.Thread(target=server.serve_forever, daemon=True), threading.Thread(target=work, daemon=True)]
    for thread in threads:
        thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer secret-key"},
    )
    try:
        try:
            with urlopen(request, timeout=60) as response:
                answer = response.status, response.headers["Content-Type"], response.read()
        except HTTPError as error:
            answer = error.code, error.headers["Content-Type"], error.read()
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        for thread in threads:
            thread.join()
    return (*answer, queue)


@pytest.mark.parametrize("stream", [False, True])
def test_context_overflow_from_the_resident_is_a_complete_error_body(tmp_path, stream):
    # Resident.prepare_api raises the engine's ApiError and the handler catches protocol.ApiError, the same class,
    # so the client gets ApiError.body() (the (ValueError, OSError) fallback would answer without param and code).
    # A stream is refused at its first chunk, before the event-stream headers: the same JSON answer.
    body = dict(model="glm-5.3", messages=[dict(role="user", content="hello")], stream=stream)
    status, kind, raw, queue = completions(tmp_path, resident(32768), body)
    assert (status, kind) == (400, "application/json; charset=utf-8")
    assert json.loads(raw) == dict(
        error=dict(
            message="this conversation fills the 32768-slot context; send less history or smaller tool output",
            type="invalid_request_error",
            param=None,
            code=None,
        )
    )
    assert queue.db["jobs"] == []


def test_a_stream_failing_after_its_headers_ends_with_an_in_band_error(tmp_path):
    # An admitted stream whose job then fails: the headers and the role chunk are out, so the error is the last
    # event before [DONE].
    backend = resident(3)
    backend.next_sequence = lambda: 771
    backend.publish = lambda job, root: None

    def observe(job):
        raise OSError("receipt unreadable")

    backend.observe = observe
    body = dict(model="glm-5.3", messages=[dict(role="user", content="hello")], stream=True)
    status, kind, raw, queue = completions(tmp_path, backend, body)
    assert (status, kind) == (200, "text/event-stream; charset=utf-8")
    events = raw.split(b"\n\n")
    assert len(events) == 4 and events[-1] == b"" and all(e.startswith(b"data: ") for e in events[:3])
    first = json.loads(events[0].removeprefix(b"data: "))
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == dict(role="assistant", content="")
    assert json.loads(events[1].removeprefix(b"data: ")) == dict(
        error=dict(message="receipt unreadable", type="server_error", param=None, code=None)
    )
    assert events[2] == b"data: [DONE]"
    assert queue.error == "receipt unreadable" and len(queue.db["jobs"]) == 1
