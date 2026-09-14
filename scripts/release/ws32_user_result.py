"""Replay a collected user response, never score it or manufacture a success seal.

The caller must authenticate original transport generations, worker ownership,
cold graph/checkpoint/topology admission and trace physical coverage separately.
This module validates actual request originals; it performs no model execution.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from glm_tpu import user_request
from scripts.greenfield.ws32_native_benchmark_result import read, replay_observation
from scripts.greenfield.ws32_native_benchmark_observability import TRACE_CAP

SCHEMA = "glm_ws32_user_replay_v1"
WORKER_PROFILE = "glm_ws32_user_worker_v1"
PHASES = {"cache_initialization", "input_transfer", "prefill_device", "prefill_request"}


def same(actual: Any, expected: Any, label: str) -> None:
    # Canonical JSON distinguishes booleans from integers, unlike Python ==.
    if user_request.canonical(actual) != user_request.canonical(expected):
        raise ValueError("user original differs: " + label)


def seconds(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("user original has invalid wall timing")
    return float(value)


def delivered_tokens(directory: Path, request: dict) -> tuple[list[int], str, str]:
    raw = read(directory / "tokens.jsonl", 32 << 20)
    if not raw.endswith(b"\n"):
        raise ValueError("user completed token stream has a partial event")
    events = [json.loads(line) for line in raw.splitlines()]
    if not 0 < len(events) <= request["max_new_tokens"]:
        raise ValueError("user token count exceeds request or is empty")
    ids = []
    for index, event in enumerate(events):
        token = event.get("token_id")
        if type(token) is not int or not 0 <= token < user_request.VOCAB:
            raise ValueError("user delivered token is invalid")
        reason = (
            "eos"
            if token in user_request.EOS
            else "length" if index + 1 == request["max_new_tokens"] else None
        )
        same(
            event,
            dict(
                request_id=request["request_id"],
                index=index,
                token_id=token,
                finish_reason=reason,
            ),
            "token frontier/stop policy",
        )
        if index < len(events) - 1 and reason is not None:
            raise ValueError("user token follows terminal event")
        ids.append(token)
    if reason is None:
        raise ValueError("user completed response lacks terminal event")
    return ids, reason, sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def trace_original(
    root: Path, rank: int, observed: dict, *, decoded: bool
) -> dict | None:
    same(observed["request_index"], 0, "observation request index")
    same(observed["request_wall_instrumented"], decoded, "instrumented request wall")
    same(
        observed["instrumented_decode_indices"],
        [0] if decoded else [],
        "instrumented indices",
    )
    same(observed["dsa_observed"], decoded, "DSA observation status")
    same(observed["traced_decode_indices"], [0] if decoded else [], "traced indices")
    trace = observed["trace"]
    if not decoded:
        same(trace, None, "prefill-only trace")
        return None
    if not isinstance(trace, dict) or type(trace.get("path")) is not str:
        raise ValueError("user decode lacks original trace")
    relative = PurePosixPath(trace["path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != f"native_trace.rank{rank}"
        or str(relative) != trace["path"]
        or not relative.name.endswith(".xplane.pb")
    ):
        raise ValueError("user trace path escaped original rank")
    same(
        {
            key: trace[key]
            for key in (
                "request_index",
                "graph",
                "actual_model_calls",
                "python_tracer_level",
            )
        },
        dict(
            request_index=0,
            graph="observer",
            actual_model_calls=1,
            python_tracer_level=0,
        ),
        "trace scope",
    )
    raw = read(root / relative, TRACE_CAP)
    same(trace["bytes"], len(raw), "trace byte count")
    same(trace["sha256"], sha256(raw).hexdigest(), "trace hash")
    return dict(rank=rank, path=str(relative), bytes=len(raw), sha256=trace["sha256"])


def replay_user_request(
    *,
    root: Path,
    request: dict,
    request_file_sha256: str,
    pin: str,
    parents: list[dict],
    tokenizer: Any,
    full_index_layers: tuple[int, ...],
) -> dict:
    """Require one terminal answer from all eight original rank directories.

    Missing/failed ranks raise, retaining originals for failure archival. No
    partial response becomes a completed response, quality score or TPU seal.
    """
    user_request.validate(request)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", pin)
        or not re.fullmatch(r"[0-9a-f]{64}", request_file_sha256)
        or len(parents) != 8
    ):
        raise ValueError(
            "user replay requires pinned source/input and eight cold parents"
        )
    directories = [root / f"sessions.rank{rank}" / "item000" for rank in range(8)]
    ids, reason, digest = delivered_tokens(directories[0], request)
    rows, traces, peaks = [], [], []
    for rank, (directory, parent) in enumerate(zip(directories, parents, strict=True)):
        if (directory / "failure.json").exists():
            raise ValueError("user response has a failure original")
        if sorted(p.name for p in directory.parent.iterdir()) != ["item000"]:
            raise ValueError("user scope contains additional requests")
        row = json.loads(read(directory / "result.json.gz", 4 << 20, compressed=True))
        outer = json.loads(read(root / f"runner.rank{rank}.json", 2 << 20))
        if "failure_type" in outer:
            raise ValueError("user worker failed after response")
        expected = dict(
            profile=WORKER_PROFILE,
            artifact_kind=WORKER_PROFILE,
            code_hash=pin,
            request_file_sha256=request_file_sha256,
            request_sha256=request["request_sha256"],
            launch_process_id=rank,
            jax_process_index=parent["jax_process_index"],
            complete=True,
            benchmark=False,
            protected_result_sealed=False,
            generated_tokens=len(ids),
            token_ids_sha256=digest,
            finish_reason=reason,
        )
        same(
            {key: outer[key] for key in expected},
            expected,
            "outer worker identity/result",
        )
        identity = json.loads(read(directory / "identity.json", 4096))
        same(
            identity,
            dict(
                index=0,
                request_id=request["request_id"],
                seed=request["seed"],
                prompt_ids_sha256=request["prompt_ids_sha256"],
                rank=rank,
            ),
            "input identity",
        )
        expected = dict(
            schema="glm_ws32_user_result_v1",
            request_id=request["request_id"],
            request_sha256=request["request_sha256"],
            rank=rank,
            complete=True,
            benchmark=False,
            quality_score=None,
            protected_result_sealed=False,
            generated_tokens=len(ids),
            token_ids_sha256=digest,
            finish_reason=reason,
            first_token_delivered_before_decode=True,
            live_session_resume=len(ids) > 1,
            delivery_boundary=(
                "nonoutput_rank" if rank else "rank0_local_jsonl_write_flush"
            ),
        )
        allowed = set(expected) | {
            "phases",
            "decode_seconds",
            "observations",
            "cold_load_compile_seconds",
            "ttft_seconds",
            "delivered_request_seconds",
        }
        if rank == 0:
            allowed.add("answer")
        if set(row) != allowed:
            raise ValueError("user response schema differs or claims benchmark fields")
        same({key: row[key] for key in expected}, expected, "fleet response/policy")
        if (
            not isinstance(row["decode_seconds"], list)
            or len(row["decode_seconds"]) != len(ids) - 1
            or set(row["phases"]) != PHASES
        ):
            raise ValueError("user phase/decode sample count differs")
        for value in [*row["decode_seconds"], *row["phases"].values()]:
            seconds(value)
        cold = seconds(row["cold_load_compile_seconds"])
        if cold != seconds(parent["cold_load_compile_seconds"]) or cold != seconds(
            outer["cold_load_compile_seconds"]
        ):
            raise ValueError("user cold wall differs from original loader")
        seconds(outer["worker_wall_seconds"])
        if rank == 0:
            if seconds(row["delivered_request_seconds"]) < seconds(row["ttft_seconds"]):
                raise ValueError("user completion precedes first delivery")
        else:
            same(
                [row["ttft_seconds"], row["delivered_request_seconds"]],
                [None, None],
                "nonoutput wall",
            )
            if (directory / "tokens.jsonl").exists() or (
                directory / "answer.txt.gz"
            ).exists():
                raise ValueError("nonoutput worker wrote user delivery")
        checked = replay_observation(
            directory,
            row,
            prompt_tokens=len(request["prompt_ids"]),
            parent=parent,
            full_index_layers=full_index_layers,
        )
        peaks.append(checked["peak"])
        trace = trace_original(root, rank, row["observations"], decoded=len(ids) > 1)
        if trace is not None:
            traces.append(trace)
        rows.append(row)
    raw_answer = read(directories[0] / "answer.txt.gz", 16 << 20)
    same(
        rows[0]["answer"],
        dict(
            name="answer.txt.gz",
            bytes=len(raw_answer),
            sha256=sha256(raw_answer).hexdigest(),
        ),
        "answer receipt",
    )
    output_ids = ids[:-1] if ids[-1] in user_request.EOS else ids
    text = tokenizer.decode(output_ids, skip_special_tokens=False).encode()
    if read(directories[0] / "answer.txt.gz", 16 << 20, compressed=True) != text:
        raise ValueError("user answer does not decode from delivered tokens")
    # First decode is the observer+trace; never count it as ordinary wall.
    ordinary = rows[0]["decode_seconds"][1:]
    total = sum(ordinary)
    return dict(
        schema=SCHEMA,
        complete=True,
        benchmark=False,
        quality_score=None,
        request_id=request["request_id"],
        request_sha256=request["request_sha256"],
        request_file_sha256=request_file_sha256,
        code_hash=pin,
        generated_tokens=len(ids),
        token_ids_sha256=digest,
        finish_reason=reason,
        prompt_tokens=len(request["prompt_ids"]),
        live_session_resume=len(ids) > 1,
        durable_resume_verified=False,
        phases=rows[0]["phases"],
        cold_load_compile_seconds=rows[0]["cold_load_compile_seconds"],
        ttft_seconds=rows[0]["ttft_seconds"],
        delivered_request_seconds=rows[0]["delivered_request_seconds"],
        delivery_boundary=rows[0]["delivery_boundary"],
        request_wall_instrumented=len(ids) > 1,
        ordinary_decode_samples=len(ordinary),
        ordinary_decode_seconds=total,
        decode_p50_ms=float(np.percentile(ordinary, 50) * 1000) if ordinary else None,
        decode_p99_ms=float(np.percentile(ordinary, 99) * 1000) if ordinary else None,
        decode_wall_tokens_per_second=len(ordinary) / total if total > 0 else None,
        maximum_peak_hbm_bytes=max(peaks),
        trace_originals=traces,
        trace_physical_coverage_verified=False,
        protected_result_sealed=False,
        worker_ownership_verified=False,
        cold_admission_verified=False,
    )
