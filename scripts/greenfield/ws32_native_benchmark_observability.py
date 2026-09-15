"""Bounded original DSA/cache/peak and one fresh sampled-observer trace.

Use already-resident programs and the SAME advancing request, never duplicate
generation, clone caches, add warmup tokens or inherit old long-run evidence.
The first decode of each request observes DSA. The first such call in the
campaign is traced; those samples are explicitly excluded from ordinary decode
wall statistics. Actual first-token delivery always precedes this instrumentation.
"""
from __future__ import annotations

from hashlib import sha256
import gzip
import io
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from glm_tpu.greenfield.validation.ws32_short_context import (
    compare_ws32_dsa_within_engine, validate_ws32_cache_probe,
)
from glm_tpu.greenfield.validation.ws32_prefill_memory import capture_identified_device_memory
from scripts.greenfield.ws32_budgeted_calls import start_device_trace, voted_trace
from scripts.greenfield.ws32_delivery_phase_evidence import _advance, _boundary
from scripts.greenfield.ws32_native_benchmark_protocol import canonical
from glm_tpu.host_paths import _plain_path

# Original XSpace includes full-model metadata (~244 MB on the first real
# sampled observer). Bound raw local/inflated bytes separately from storage.
TRACE_CAP = 320 << 20
TRACE_STORED_CAP = 128 << 20
WITNESS_CAP = 2 << 20


def trace_storage_bytes(path: Path) -> int:
    """Count lossless gzip bytes without another full trace copy.

    Same level/header geometry as publish_exact; actual published size is also
    checked. Already-compressed JSON remains unchanged. No events are removed.
    """
    if not path.name.endswith(".xplane.pb"):
        return path.stat().st_size
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            size += len(compressor.compress(block))
    return size + len(compressor.flush())


def npz_bytes(**arrays: Any) -> bytes:
    if sum(np.asarray(v).nbytes for v in arrays.values()) > WITNESS_CAP:
        raise ValueError("native compact witness exceeds raw budget")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    if stream.tell() > WITNESS_CAP:
        raise ValueError("native compact witness exceeds compressed budget")
    return stream.getvalue()


class NativeObservability:
    def __init__(self, loaded: Any, store: Any):
        self.loaded, self.store = loaded, store
        self.runtime = loaded.runtime
        self.trace_record = None
        self.traced_request = None
        self.current = None
        self.slots = {r["device_id"]: r["slot"] for r in loaded.record["local_slots"]}
        self.process = loaded.record["jax_process_index"]

    def begin(self, index: int, session: Any) -> None:
        self.current = dict(request_index=index, dsa_observed=False,
                            instrumented_decode_indices=[], traced_decode_indices=[])
        underlying = session._decode
        count = 0
        def decode(token, state, uniform):
            nonlocal count
            this = count
            count += 1
            if this != 0:
                return underlying(token, state, uniform)
            extra = (self.runtime.exact_weights,) if self.runtime.decode_config.exact_dsa else ()
            def execute():
                value = self.loaded.compiled["observer"](token, state,
                    self.runtime.decode_weights, *extra, self.runtime.rope, uniform)
                jax.block_until_ready(value)
                return value
            if self.trace_record is None:
                trace = self.store.root.parent / f"native_trace.rank{self.store.rank}"
                adapter = SimpleNamespace(phase=lambda _name, fn:self.runtime._phase(fn))
                with voted_trace(adapter, trace, start=start_device_trace, stop=jax.profiler.stop_trace):
                    value = execute()
                # Preserve the completed model observation even if trace
                # finalization/publication refuses. This remains outside trace.
                self.runtime._phase(lambda:self.observe_dsa(value.dsa, session.policy.prompt_tokens))
                self.runtime._phase(lambda:self.finish_trace(trace, index))
                self.current["traced_decode_indices"].append(this)
            else:
                value = execute()
                self.runtime._phase(lambda:self.observe_dsa(value.dsa, session.policy.prompt_tokens))
            self.current["instrumented_decode_indices"].append(this)
            return value.result
        session._decode = decode

    def finish_trace(self, root: Path, index: int) -> None:
        files = []
        for path in root.rglob("*"):
            _plain_path(path)
            if path.is_file():
                files.append(path)
        if sum(p.stat().st_size for p in files) > TRACE_CAP:
            raise ValueError("native trace exceeds complete-rank budget")
        xplanes = [p for p in files if p.name.endswith(".xplane.pb")]
        if len(xplanes) != 1 or not 0 < xplanes[0].stat().st_size <= TRACE_CAP:
            raise ValueError("native trace requires one bounded original XPlane")
        if sum(trace_storage_bytes(p) for p in files) > TRACE_STORED_CAP:
            raise ValueError("native trace exceeds compressed storage budget")
        path = xplanes[0]
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1 << 20), b""):
                digest.update(chunk)
        self.trace_record = dict(path=str(path.relative_to(self.store.root.parent)),
            bytes=path.stat().st_size, sha256=digest.hexdigest(), request_index=index,
            graph="observer", actual_model_calls=1, python_tracer_level=0,
            scope="sampled observer, not pure decode latency or prefill attribution")
        self.traced_request = index

    def observe_dsa(self, value: Any, position: int) -> None:
        host = jax.device_get(value)
        arrays = dict(producer_layer_ids=np.asarray(host.producer_layer_ids, np.int32),
            selected_positions=np.asarray(host.selected_positions, np.int32),
            selected_valid_counts=np.asarray(host.selected_valid_counts, np.int32),
            selected_scores=np.asarray(host.selected_scores, np.float32))
        self.store.save("dsa.npz", npz_bytes(**arrays), WITNESS_CAP)
        expected = np.array(tuple(self.runtime.decode_config.full_index_slots), np.int32)
        result = compare_ws32_dsa_within_engine(**arrays, decode_position=position,
            step=0, expected_producer_layer_ids=expected)
        if (arrays["selected_positions"].shape != (len(expected), 1, 2048)
                or not np.all(arrays["selected_valid_counts"] == min(position+1, 2048))):
            raise ValueError("native complete DSA count/producer geometry differs")
        self.current["dsa"] = result
        if not result["passed"]:
            raise ValueError("native executing DSA selection order/ties failed")
        self.current["dsa_observed"] = True

    def finish(self, session: Any) -> dict:
        def validate():
            config = self.runtime.decode_config
            result = self.loaded.compiled["cache_probe"](session._state)
            jax.block_until_ready(result)
            host = jax.device_get(result)
            arrays = dict(position=np.asarray(host.position),
                kv_bfloat16_bits=np.ascontiguousarray(np.asarray(host.kv_rows)).view(np.uint16),
                index_bfloat16_bits=np.ascontiguousarray(np.asarray(host.index_rows)).view(np.uint16),
                contract_valid=np.asarray(host.contract_valid))
            self.store.save("cache.npz", npz_bytes(**arrays), WITNESS_CAP)
            cache = validate_ws32_cache_probe(position=np.asarray(host.position),
                kv_rows=np.asarray(host.kv_rows), index_rows=np.asarray(host.index_rows),
                contract_valid=np.asarray(host.contract_valid),
                expected_position=session.policy.prompt_tokens+len(session.events)-2,
                num_layers=config.geometry.num_layers, full_indexer_count=len(config.full_index_slots),
                packed_cache_width=config.packed_cache_width, index_width=config.geometry.dsa_indexer_head_dim)
            self.current["cache"] = cache
            if not cache["passed"]:
                raise ValueError("native final cache write witness failed")
            post = list(capture_identified_device_memory(tuple(jax.local_devices())))
            self.store.save("final_memory.json", canonical(post), 64 << 10)
            import json
            before = json.loads(gzip.decompress((self.store.current / "prefill_done.json.gz").read_bytes()))
            _advance(_boundary(before["census"]["devices"], self.slots, self.process, nested=True),
                     _boundary(post, self.slots, self.process, nested=False))
            if len(session.events) > 1 and not self.current["dsa_observed"]:
                raise ValueError("native decode lacks its original DSA observation")
            self.current["trace"] = self.trace_record if self.traced_request == self.current["request_index"] else None
            self.current["request_wall_instrumented"] = bool(self.current["instrumented_decode_indices"])
            return dict(self.current)
        return self.runtime._phase(validate)
