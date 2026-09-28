"""The saved conversations of the browser chat UI, kept in the job queue's document.

A conversation is a titled list of messages. Sending a message queues a job on the shared job queue
(``glm_tpu.entrypoints.serve.job_queue``) with the conversation's whole history; the queue runs it and hands the
job's progress back here. Conversations and jobs stay one document (``chats.json``) under the queue's one lock.
"""

import json
import re
import time
import uuid

from glm_tpu.entrypoints.serve.job_queue import PENDING


class ConversationStore:
    """Create, rename, send to and delete conversations, and the browser's view of them, over ``queue``."""

    def __init__(self, queue):
        self.queue = queue
        queue.conversations = self

    def snapshot(self):
        with self.queue.mutex:
            availability = self.queue.error
            try:
                self.queue.backend.check()
            except (ValueError, OSError) as exc:
                availability = str(exc)
            # Exclude private model payloads from browser responses.
            jobs = [{k: v for k, v in j.items() if k != "payload"} for j in self.queue.db["jobs"] if not j.get("api")]
            return json.loads(
                json.dumps(
                    dict(
                        chats=self.queue.db["chats"],
                        jobs=jobs,
                        error=availability,
                        model="GLM-5.3",
                        capacity=self.queue.backend.capacity,
                        scheduling="sequential",
                    )
                )
            )

    def change(self, data):
        with self.queue.mutex:
            action = data.get("action")
            if action == "create":
                chat = dict(id=uuid.uuid4().hex, title="New conversation", messages=[])
                self.queue.db["chats"].append(chat)
                self.queue.save()
                return dict(id=chat["id"])
            chat = next((c for c in self.queue.db["chats"] if c["id"] == data.get("chat")), None)
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
                existing = next((j for j in self.queue.db["jobs"] if j["id"] == key), None)
                if existing:
                    if existing["chat"] != chat["id"] or existing["user_text"] != text:
                        raise ValueError("Message identity was already used.")
                    return dict(id=key)
                if type(text) is not str or not text.strip() or len(text.encode()) > 128000:
                    raise ValueError("Enter a message of at most 128,000 bytes.")
                if self.queue.error:
                    raise ValueError(self.queue.error)
                if len([j for j in self.queue.db["jobs"] if j["status"] in ("queued", "generating")]) >= PENDING:
                    raise ValueError("Ten messages are already waiting. Wait for an answer.")
                if chat["messages"] and chat["messages"][-1].get("status") != "complete":
                    raise ValueError("Wait for a completed answer, or start a new conversation.")
                messages = [dict(role=m["role"], content=m["content"]) for m in chat["messages"]]
                messages.append(dict(role="user", content=text))
                payload = self.queue.backend.prepare(messages, key)
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
                self.queue.db["jobs"].append(job)
                chat["messages"].extend(
                    [dict(role="user", content=text), dict(role="assistant", content="", status="queued", job=key)]
                )
                if len(chat["messages"]) == 2 and chat["title"] == "New conversation":
                    chat["title"] = " ".join(text.split())[:55]
            elif action == "delete":
                if any(
                    j["chat"] == chat["id"] and j["status"] in ("queued", "generating") for j in self.queue.db["jobs"]
                ):
                    raise ValueError("Wait for this conversation to finish before deleting it.")
                self.queue.db["chats"].remove(chat)
                # Original inference evidence stays in the private resident run.
                self.queue.db["jobs"] = [j for j in self.queue.db["jobs"] if j["chat"] != chat["id"]]
            else:
                raise ValueError("Unknown action.")
            self.queue.save()
            return dict(ok=True)

    def update_message(self, job):
        """Copy a conversation job's answer and status into its assistant message (``JobQueue.step`` calls this
        under the queue's lock and saves the document afterwards)."""
        chat = next(c for c in self.queue.db["chats"] if c["id"] == job["chat"])
        message = next(m for m in chat["messages"] if m.get("job") == job["id"])
        message.update(content=job["answer"], status=job["status"])
