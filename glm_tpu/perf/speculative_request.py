"""Guarded greedy MTP-assisted delivery on already admitted executables.

All graphs, weights and caches are owned by the worker. Accepted output alone
counts toward throughput. No sampling, automatic retry or crash recovery.
"""
from dataclasses import dataclass
import math

import jax
import numpy as np

from ..greenfield.runtime.ws32_request_session import Ws32RequestSession, TokenEvent
from .mtp_state import TargetHistory


@dataclass(frozen=True)
class Acceptance:
    count: int
    accepted_drafts: int
    reason: str | None


def accept_greedy(inputs, predictions, remaining, eos_ids):
    """Host equivalent of speculative_accept.greedy_acceptance, without dispatch."""
    if (inputs.ndim != 1 or not 1 <= inputs.size <= 3
            or predictions.shape != inputs.shape or inputs.dtype != np.int32
            or predictions.dtype != np.int32 or type(remaining) is not int or remaining < 1
            or np.any(inputs < 0) or np.any(predictions < 0)):
        raise ValueError('invalid greedy speculative token plan')
    accepted = 0
    for i in range(inputs.size - 1):
        if int(inputs[i+1]) != int(predictions[i]):
            break
        accepted += 1
    count = min(accepted + 1, remaining)
    reason = 'length' if count == remaining else None
    for i in range(count):
        if int(predictions[i]) in eos_ids:
            count, reason = i+1, 'eos'
            break
    return Acceptance(count, min(accepted, count), reason)


class SpeculativeRequestSession(Ws32RequestSession):
    """One pending target token, a committed native cache, and 1..3 target rows.

    propose(pending,target,native,rows) returns (input_ids,target_proposal).
    commit(target,proposal,count) and refresh(native,history,count) are compiled
    device callbacks. fleet_agree must compare the entire int32 vector across
    all authenticated hosts, not merely vote on local validity. It runs after
    local plan admission and before either cache root can be committed.
    """

    def __init__(self, *args, native_state, propose, commit, refresh, replicate_count,
                 fleet_agree, rows=3, **kwargs):
        if type(rows) is not int or rows not in (2, 3):
            raise ValueError('only qualified two/three-row verifier families are allowed')
        super().__init__(*args, **kwargs)
        self._native = native_state
        self._propose = propose
        self._commit = commit
        self._refresh = refresh
        self._count = replicate_count
        self._agree = fleet_agree
        self.rows = rows
        self.rounds = []

    def step(self):
        self._begin()
        try:
            if not self._events or self._native is None:
                raise RuntimeError('prefill and native bootstrap must complete before speculation')
            started = self._clock()
            remaining = self.policy.max_new_tokens - len(self._events)
            rows = min(self.rows, remaining)
            position = self.policy.prompt_tokens + len(self._events) - 1
            error = None
            try:
                valid = (bool(np.asarray(self._native.cache.contract_valid).all())
                    and np.array_equal(np.asarray(self._native.cache.position), [position])
                    and np.array_equal(np.asarray(self._native.cache.context_lengths), [position+1]))
            except Exception as exc:
                valid, error = False, exc
            self._vote(valid, error)
            before = self._clock()
            inputs, proposal = self._propose(self._pending_token, self._state, self._native, rows)
            jax.block_until_ready((inputs, proposal))
            proposal_seconds = self._clock()-before
            error = None
            try:
                ids, predictions = np.asarray(inputs), np.asarray(proposal.predictions)
                plan = accept_greedy(ids, predictions, remaining, self.policy.eos_ids)
                valid = (ids.size == rows and int(ids[0]) == self._events[-1].token_id
                    and bool(np.asarray(proposal.contract_valid).all())
                    and np.all(ids < self.policy.vocab_size)
                    and np.all(predictions < self.policy.vocab_size)
                    and math.isfinite(proposal_seconds) and proposal_seconds >= 0)
            except Exception as exc:
                valid, error = False, exc
            self._vote(valid, error)
            # Same shape on all hosts because the output frontier is agreed.
            metadata = np.concatenate((np.asarray([position, remaining, plan.count,
                plan.accepted_drafts], np.int32), ids, predictions))
            agreed = self._agree(metadata)
            if type(agreed) is not bool or not agreed:
                raise RuntimeError('speculative token plan differs across hosts')
            count = self._count(np.asarray(plan.count, np.int32))
            before = self._clock()
            target = self._commit(self._state, proposal, count)
            # Deliberately refresh from the old committed root, never the
            # speculative recurrent cache. Shifted IDs come from the target.
            history = TargetHistory(self._state.position, proposal.predictions,
                proposal.normalized_hidden_local, proposal.contract_valid)
            native = self._refresh(self._native, history, count)
            jax.block_until_ready((target, native))
            refresh_commit_seconds = self._clock()-before
            error = None
            try:
                valid = all(bool(np.asarray(s.contract_valid).all())
                    and np.array_equal(np.asarray(s.position), [position+plan.count])
                    and np.array_equal(np.asarray(s.context_lengths), [position+plan.count+1])
                    for s in (target, native.cache))
                valid &= (math.isfinite(refresh_commit_seconds) and refresh_commit_seconds >= 0
                    and np.asarray(native.next_token).shape == (1,)
                    and np.asarray(native.next_token).dtype == np.int32
                    and 0 <= int(np.asarray(native.next_token)[0]) < self.policy.vocab_size)
            except Exception as exc:
                valid, error = False, exc
            self._vote(valid, error)
            # Both roots commit before any sink invocation. An ambiguous
            # partial batch delivery poisons the request and cannot be replayed.
            self._state, self._native = target, native
            self._pending_token = proposal.predictions[plan.count-1:plan.count]
            delivered_events = []
            for i in range(plan.count):
                event = TokenEvent(self.policy.request_id, len(self._events), int(predictions[i]),
                                   plan.reason if i == plan.count-1 else None)
                self._events.append(event)
                try:
                    self._deliver(event)
                    now = self._clock()
                    floor = self._delivered_at[-1] if self._delivered_at else self.request_started
                    if not math.isfinite(now) or now < floor:
                        raise ValueError('delivery clock moved backwards')
                    self._delivered_at.append(now)
                    delivered_events.append(event)
                except Exception as exc:
                    error = exc
                    break
            self._vote(error is None, error)
            elapsed = self._clock()-started
            self._decode_seconds.append(proposal_seconds+refresh_commit_seconds)
            self.rounds.append(dict(rows=rows,emitted=plan.count,accepted_drafts=plan.accepted_drafts,
                proposed_drafts=rows-1,proposal_seconds=proposal_seconds,
                refresh_commit_seconds=refresh_commit_seconds,wall_seconds=elapsed))
            return tuple(delivered_events)
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False

    def release(self):
        super().release()
        self._native = None
