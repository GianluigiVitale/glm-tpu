"""Bounded dense-only correction: four WK calls and one first128 candidate.

Not a launcher, checkpoint loader, compiler admission or decoder promotion.
The parent must authenticate DB605 originals, DB604 context, selected weights,
physical owners and this distinct profile before calling the continuation.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import ws32_dense_frontier_prepare as preparation
from scripts.greenfield import ws32_dense_frontier_worker as original
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.prefill_layer_numerical import FIELDS
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution

PROTOCOL = "ws32-dense01-canonical-placement-db605-narrow-v1"
GRAPH = "dense01_canonical"
PROGRAMS = ("wk_decode", "wk_promote", GRAPH)
RAW = {
    "wk_decode": (
        10725,
        "8eeefbb0cbc3518ac223b49e1bade70dc1aa78c965c983ebef83c0140284c362",
    ),
    "wk_promote": (
        802,
        "7b277bb821af372bd03687010b1db3630533dc9db46747863dd08cc0742006e5",
    ),
    GRAPH: (667699, "d17cbfeaa7a173f5872632c16e4fbcd1de7f73898f0ca96d6d69734b32a41dad"),
}
CALLS = tuple(
    (f"layer{layer}/{name}", name) for layer in (0, 1) for name in PROGRAMS[:2]
) + (("canonical/first128", GRAPH),)
CAPSULE = "canonical_first128"
NARROW = ("narrow_32", "narrow_64", "narrow_96", "narrow_128")
MODEL_SOURCE_OVERRIDES = {
    "glm_tpu/greenfield/kernels/ws32_prefill_window.py": "87f6fcad74aa2bbaa0e1ef7309c768021eb6d9686a5ff767a1776fc81a6c8c5d",
    "glm_tpu/greenfield/kernels/ws32_prefill_dense_canonical.py": "513635cc9c792f94792841041e6b3b89df3dbb0725ae5506c34ba2d1df198436",
}


def require_source(repo: Path) -> None:
    """Fixed two-file correction; every other model source stays frozen.

    Metadata preparation only, not acquired-HLO or full-decoder authorization.
    The protected parent additionally binds a clean published pin. Historical
    numerical profiles retain their original strict source checks.
    """
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        FROZEN_SOURCE_PIN,
        MODEL_SOURCE,
    )

    for name, expected in MODEL_SOURCE_OVERRIDES.items():
        path = repo / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("canonical dense source differs from registration")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            FROZEN_SOURCE_PIN,
            "--",
            *MODEL_SOURCE,
            *(":(exclude)" + name for name in MODEL_SOURCE_OVERRIDES),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("canonical dense changed an unrelated frozen model source")


def prepare(mesh: Any, *, repo: Path) -> preparation.PreparedDenseFrontier:
    """Reuse final-layout metadata, original input/spec trees, and two-layer body."""
    return preparation.prepare(mesh, repo=repo, canonical_dense=True)


def compiler_programs(prepared: preparation.PreparedDenseFrontier, mesh: Any) -> tuple:
    jobs = preparation.compiler_programs(prepared, mesh)
    return (*jobs[:2], (GRAPH, prepared.program, prepared.inputs))


def memory_budget(census: Any, analyses: Any, *, active_graph: str) -> dict:
    if set(analyses) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("canonical dense requires its three resident programs")
    return budget_resident_execution(
        census,
        analyses,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=original.RESERVE,
    )


def narrow_target(originals: Mapping, slots: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Assemble saved narrow rows and ONLY the final saved endpoint caches.

    BF16 stays uint16 throughout, including signed-zero and payload bits.
    These arrays are not recomputed references. The caller authenticates source
    generations/hashes with the existing DB605 loader before using this mapper.
    Every original full128 health mask must be healthy, including padding.
    """
    if (
        len(slots) != 4
        or len(set(slots)) != 4
        or any(type(s) is not int or not 0 <= s < 32 for s in slots)
        or not set(NARROW) <= set(originals)
    ):
        raise ValueError("canonical retained narrow owner/capsule inventory differs")
    cache_fields = {"kv", "index", "repair"}
    endpoint_keys = {
        f"slot{s}_layer{layer}__{field}"
        for s in slots
        for layer in (0, 1)
        for field in FIELDS
    }
    row_keys = {
        key for key in endpoint_keys if key.rsplit("__", 1)[1] not in cache_fields
    }
    for label in NARROW:
        if set(originals[label]) != (
            endpoint_keys if label == NARROW[-1] else row_keys
        ):
            raise ValueError("canonical retained narrow field inventory differs")
    result = {}
    for key in sorted(endpoint_keys):
        field = key.rsplit("__", 1)[1]
        if field in cache_fields:
            value = originals[NARROW[-1]][key]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.uint16
                or value.ndim != 3
            ):
                raise ValueError("canonical retained cache encoding differs")
            result[key] = value
            continue
        parts = [originals[label][key] for label in NARROW]
        for value in parts:
            if (
                not isinstance(value, np.ndarray)
                or value.ndim < 1
                or value.shape[0] != (128 if field == "health" else 32)
                or value.shape != parts[0].shape
                or value.dtype != parts[0].dtype
            ):
                raise ValueError("canonical retained narrow row geometry differs")
            if field == "health" and (
                value.dtype != np.bool_ or value.ndim != 1 or not value.all()
            ):
                raise ValueError("canonical retained full health failed")
        result[key] = np.concatenate([value[:32] for value in parts], axis=0)
    return result


def compare(arrays: Mapping, originals: Mapping, slots: tuple[int, ...]) -> dict:
    """Exact all-field comparison including full candidate health and endcaches."""
    # The retained reader also imports execution journal types. Defer this
    # dependency so fresh inspector-first processes do not form an import cycle.
    from scripts.greenfield.ws32_dense_norm_originals import require_reproduction

    target = narrow_target(originals, slots)
    report = require_reproduction(arrays, target)
    return {
        **report,
        "scope": "DB605_NARROW_ROWS_FULL_HEALTH_AND_DENSE01_ENDPOINT_CACHES",
        "slots": sorted(slots),
        "model_calls": 1,
        "wk_calls": 4,
        "token11_cause_proven": False,
        "performance_claim": False,
    }


def execute_after_wk(
    calls: BudgetedCalls,
    *,
    mesh: Any,
    config: Any,
    prompt_tokens: np.ndarray,
    embedding: Any,
    layers: Any,
    wk: Any,
    rope: Any,
    originals: Mapping,
) -> None:
    """Exactly one candidate, archived before any health/reproduction refusal.

    Retained originals must already be generation/CRC/SHA and context-bound by
    the parent. No old wide/narrow model call or additional boundary capture.
    """

    def preflight():
        original.require_config(config)
        if (
            calls.budgeter is not memory_budget
            or calls.record.get("protocol") != PROTOCOL
            or set(calls.programs) != set(PROGRAMS)
            or [(v["phase"], v["graph"]) for v in calls.record["call_evidence"]]
            != list(CALLS[:4])
            or not all(v["completed"] for v in calls.record["call_evidence"])
            or not isinstance(prompt_tokens, np.ndarray)
            or prompt_tokens.dtype != np.int32
            or prompt_tokens.shape != (8155,)
            or sha256(prompt_tokens.tobytes()).hexdigest() != original.PROMPT_SHA
        ):
            raise ValueError("canonical prompt/budget/WK call identity differs")
        # Validate reference geometry before any model dispatch; discard assembled
        # row copies here so they are not held across the device-memory census.
        narrow_target(originals, tuple(calls.local_slots.values()))

    calls.phase("canonical/preflight", preflight)
    result_record = dict(
        complete=False, numerical_promotion=False, performance_claim=False
    )
    calls.record["dense_canonical"] = result_record

    def preserve(result):
        arrays, report = original.capture(
            result,
            local_slots=calls.local_slots,
            process_index=calls.record["jax_process_index"],
            count=128,
            keep_caches=True,
        )
        amount = sum(v.nbytes for v in arrays.values())
        if amount + (1 << 20) > original.ORIGINALS_LIMIT:
            raise ValueError("canonical originals exceed fixed rank budget")
        path = calls.root / (CAPSULE + ".npz")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        report.update(
            npz_sha256=save_arrays(path, arrays),
            npz_bytes=path.stat().st_size,
            raw_array_bytes=amount,
        )
        result_record["original"] = report
        _atomic_json(calls.root / (CAPSULE + ".json"), report)
        if not report["valid"]:
            raise ValueError("canonical output unhealthy; originals preserved")

    caches = calls.phase("canonical/initial", lambda: original.fresh_caches(mesh))
    values = calls.phase(
        "canonical/inputs",
        lambda: original.inputs(
            mesh, prompt_tokens[:128], 0, caches, embedding, layers, wk, rope
        ),
    )
    result = calls.call(*CALLS[-1], values, preserve=preserve)
    del result, values, caches

    def reproduce():
        with np.load(calls.root / (CAPSULE + ".npz"), allow_pickle=False) as saved:
            report = compare(saved, originals, tuple(calls.local_slots.values()))
        result_record.update(complete=True, comparison=report)
        _atomic_json(calls.root / "comparison.json", report)

    calls.phase("canonical/reproduction", reproduce)
