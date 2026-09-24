"""The persistent queue of chat jobs, served one at a time by the resident controller."""

import json
import re
import threading
import time
import uuid

from glm_tpu.entrypoints.openai import protocol
from glm_tpu.utils.io_utils import atomic


PENDING = 10
API_HISTORY = 50


class Chats:
    def __init__(self, root, backend):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_mode & 0o077:
            raise ValueError("Chat directory must be private (mode 0700).")
        self.root, self.backend = root, backend
        self.mutex = threading.RLock()
        self.db = (
            json.loads((root / "chats.json").read_text()) if (root / "chats.json").exists() else dict(chats=[], jobs=[])
        )
        self.error = None
        self.stopping = threading.Event()

    def save(self):
        atomic(self.root / "chats.json", self.db)

    def snapshot(self):
        with self.mutex:
            availability = self.error
            try:
                self.backend.check()
            except (ValueError, OSError) as exc:
                availability = str(exc)
            # Exclude private model payloads from browser responses.
            jobs = [{k: v for k, v in j.items() if k != "payload"} for j in self.db["jobs"] if not j.get("api")]
            return json.loads(
                json.dumps(
                    dict(
                        chats=self.db["chats"],
                        jobs=jobs,
                        error=availability,
                        model="GLM-5.3",
                        capacity=32768,
                        scheduling="sequential",
                    )
                )
            )

    def change(self, data):
        with self.mutex:
            action = data.get("action")
            if action == "create":
                chat = dict(id=uuid.uuid4().hex, title="New conversation", messages=[])
                self.db["chats"].append(chat)
                self.save()
                return dict(id=chat["id"])
            chat = next((c for c in self.db["chats"] if c["id"] == data.get("chat")), None)
            if chat is None:
                raise ValueError("Conversation not found.")
            if action == "rename":
                title = data.get("title")
                if type(title) is not str or not 1 <= len(title.strip()) <= 100:
                    raise ValueError("Use a title between 1 and 100 characters.")
                chat["title"] = title.strip()
            elif action == "send":
                key = data.get("id")
                text = data.get("text")
                if type(key) is not str or not re.fullmatch("[a-f0-9]{32}", key):
                    raise ValueError("Invalid message identity.")
                existing = next((j for j in self.db["jobs"] if j["id"] == key), None)
                if existing:
                    if existing["chat"] != chat["id"] or existing["user_text"] != text:
                        raise ValueError("Message identity was already used.")
                    return dict(id=key)
                if type(text) is not str or not text.strip() or len(text.encode()) > 128000:
                    raise ValueError("Enter a message of at most 128,000 bytes.")
                if self.error:
                    raise ValueError(self.error)
                if len([j for j in self.db["jobs"] if j["status"] in ("queued", "generating")]) >= PENDING:
                    raise ValueError("Ten messages are already waiting. Wait for an answer.")
                if chat["messages"] and chat["messages"][-1].get("status") != "complete":
                    raise ValueError("Wait for a completed answer, or start a new conversation.")
                messages = [dict(role=m["role"], content=m["content"]) for m in chat["messages"]]
                messages.append(dict(role="user", content=text))
                payload = self.backend.prepare(messages, key)
                job = dict(
                    id=key,
                    chat=chat["id"],
                    user_text=text,
                    status="queued",
                    payload=payload,
                    sequence=None,
                    answer="",
                    thinking="",
                    output_tokens=0,
                    prompt_tokens=len(payload["prompt_ids"]),
                    created=time.time(),
                )
                self.db["jobs"].append(job)
                chat["messages"].extend(
                    [dict(role="user", content=text), dict(role="assistant", content="", status="queued", job=key)]
                )
                if len(chat["messages"]) == 2 and chat["title"] == "New conversation":
                    chat["title"] = " ".join(text.split())[:55]
            elif action == "delete":
                if any(j["chat"] == chat["id"] and j["status"] in ("queued", "generating") for j in self.db["jobs"]):
                    raise ValueError("Wait for this conversation to finish before deleting it.")
                self.db["chats"].remove(chat)
                # Original inference evidence stays in the private resident run.
                self.db["jobs"] = [j for j in self.db["jobs"] if j["chat"] != chat["id"]]
            else:
                raise ValueError("Unknown action.")
            self.save()
            return dict(ok=True)

    def submit(self, payload, key, *, label):
        """Queue a stateless request; it belongs to no saved conversation."""
        with self.mutex:
            existing = next((j for j in self.db["jobs"] if j["id"] == key), None)
            if existing is not None:
                return existing
            if self.error:
                raise protocol.ApiError(self.error, status=503, kind="server_error")
            if len([j for j in self.db["jobs"] if j["status"] in ("queued", "generating")]) >= PENDING:
                raise protocol.ApiError(
                    "too many requests are already waiting for this one model", status=429, kind="rate_limit_error"
                )
            job = dict(
                id=key,
                chat=None,
                api=True,
                label=label,
                user_text="",
                status="queued",
                payload=payload,
                sequence=None,
                answer="",
                thinking="",
                output_tokens=0,
                prompt_tokens=len(payload["prompt_ids"]),
                created=time.time(),
            )
            self.db["jobs"].append(job)
            self.retire()
            self.save()
            return job

    def job(self, key):
        with self.mutex:
            found = next((j for j in self.db["jobs"] if j["id"] == key), None)
            return None if found is None else json.loads(json.dumps({k: v for k, v in found.items() if k != "payload"}))

    def retire(self):
        """Bound the stateless history; saved conversations are never touched."""
        finished = [j for j in self.db["jobs"] if j.get("api") and j["status"] in ("complete", "incomplete")]
        for job in finished[:-API_HISTORY] if len(finished) > API_HISTORY else []:
            self.db["jobs"].remove(job)

    def step(self):
        with self.mutex:
            job = next((j for j in self.db["jobs"] if j["status"] in ("queued", "generating")), None)
            if job is None or self.error:
                return
            try:
                if job["sequence"] is None:
                    job["sequence"] = self.backend.next_sequence()
                    self.save()  # Durable identity before any inbox write.
                self.backend.publish(job, self.root)
                update = self.backend.observe(job)
                job.update(update)
                if job.get("chat"):
                    chat = next(c for c in self.db["chats"] if c["id"] == job["chat"])
                    message = next(m for m in chat["messages"] if m.get("job") == job["id"])
                    message.update(content=job["answer"], status=job["status"])
                self.save()
            except Exception as exc:
                # Preserve admission identity. Restart can reconcile the SAME request;
                # never resubmit under a fresh sequence after an uncertain failure.
                self.error = str(exc)
                job["error"] = self.error
                atomic(self.root / "error.json", dict(error=self.error, job=job["id"], time=time.time()))

    def work(self):
        while not self.stopping.is_set():
            self.step()
            with self.mutex:
                active = any(j["status"] == "generating" for j in self.db["jobs"])
            self.stopping.wait(0.25 if active else 1)
