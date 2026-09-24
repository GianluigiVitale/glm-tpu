"""Host ownership and delivery for one fixed batch of independent conversations."""
import time

import jax
import numpy as np

from glm_tpu.engine.request_session import TokenEvent


class BatchedSession:
    def __init__(self, requests, *, decode, put, vote, deliver, deadline,
                 clock=time.perf_counter):
        self.requests=requests
        self.decode,self.put,self.vote,self.deliver=decode,put,vote,deliver
        self.deadline,self.clock=deadline,clock
        self.events=[[] for _ in requests]
        self.active=np.ones(len(requests),bool)
        self.finished_at=[None]*len(requests)
        self.failed=False
        self.round=0

    def require(self, valid):
        agreed=self.vote(bool(valid) and self.clock()<self.deadline)
        if not valid or agreed is not True:
            raise RuntimeError('concurrent batch rejected state, delivery or deadline')

    def accept(self, metadata):
        """Admit the whole round before committing/delivering any of its tokens."""
        status=np.asarray(metadata)
        valid=status.shape==(len(self.requests),4) and status.dtype==np.int32
        if valid:
            for lane,item in enumerate(self.requests):
                if not self.active[lane]:continue
                token,health,position,length=map(int,status[lane])
                expected=len(item['prompt_ids'])+len(self.events[lane])
                valid &= (0<=token<item['vocab_size'] and health==1
                          and position==expected and length==expected+1)
        self.require(valid)
        error=None
        try:
            for lane,item in enumerate(self.requests):
                if not self.active[lane]:continue
                token=int(status[lane,0])
                reason=('eos' if token in item['eos_ids'] else 'length'
                        if len(self.events[lane])+1==item['max_new_tokens'] else None)
                event=TokenEvent(item['request_id'],len(self.events[lane]),token,reason)
                self.events[lane].append(event)
                if reason:self.active[lane]=False
                self.deliver(lane,event,self.round)
                if reason:self.finished_at[lane]=self.clock()
        except Exception as exc:error=exc
        self.require(error is None)
        if error is not None:raise RuntimeError('concurrent delivery failed') from error

    def run(self, state, tokens, first_metadata):
        if self.failed or self.round or any(self.events):
            raise RuntimeError('concurrent session cannot be reused')
        try:
            self.accept(first_metadata)
            self.decode_started=self.clock()
            while self.active.any():
                self.require(True)
                out=jax.block_until_ready(self.decode(tokens,state,self.put(self.active)))
                # Transfer exclusive ownership before delivery: an ambiguous
                # failure poisons this session and cannot replay consumed state.
                state,tokens=out.state,out.next_token
                self.round+=1
                self.accept(out.metadata)
                del out
            self.decode_seconds=self.clock()-self.decode_started
        except Exception:
            self.failed=True
            raise
        finally:
            # No live cache survives the completed or failed batch.
            del state,tokens
