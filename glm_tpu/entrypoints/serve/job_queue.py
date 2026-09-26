"""The persistent queue of chat jobs, served one at a time by the resident controller."""

import json
import threading
import time

from glm_tpu.entrypoints.openai import protocol
from glm_tpu.utils.io_utils import atomic


PENDING = 10
API_HISTORY = 50


class JobQueue:
    """The jobs of the browser workspace and the ``/v1`` API, run one at a time against the resident controller.

    It owns the private state directory: ``chats.json`` (saved conversations and jobs, one document) and the lock
    that guards it. The conversation store (``glm_tpu.entrypoints.ui.conversations``) edits the conversations in
    the same document under the same lock, and ``step`` hands it each conversation job's progress.
    """

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
        self.conversations = None  # set by the ConversationStore built on this queue

    def save(self):
        atomic(self.root / "chats.json", self.db)

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
                    self.conversations.update_message(job)
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
