"""Fixed two-branch layers0..6 history identity; never execution authorization.

The candidate branch is the corrected canonical B128/B114 grouping whose own8K
run failed token11; the control is the retained live32 grouping that passed.
Both replay the ORIGINAL 8155-token prompt through layers0..6 with separate
caches, then the exact first-decode observer runs on each branch's completed
REPAIRED history. Each branch's observer must reproduce its own retained step0
event0..3 arrays before any boundary difference may be read as attribution.
No model arithmetic, checkpoint, numerical bound or performance target changes.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, NamedTuple

import numpy as np

from scripts.greenfield.ws32_history_frontier import LAYERS, PRODUCERS

PROTOCOL = "ws32-history-frontier-l06-two-branch-first-decode-v1"
KERNEL = "ws32_history_frontier"
SELECTED_LEAVES = 201
PAYLOAD_BYTES = 1424692176
OVERLAY_TENSORS = 12
OVERLAY_BYTES = 65691648
CAPACITY = 8192
PROMPT_LENGTH = 8155
# The sealed original 8K prompt; identical to the dense diagnostics' pin.
PROMPT_SHA = "d860b7f4be91608c86e0a629c4096fd7a95036287d0f8e31ea67b01475de0cc0"
WITNESS = dict(token=220, position=8155, context_length=8156)
BRANCHES = ("candidate", "control")
WIDE, NARROW, TAIL_ROWS = 128, 32, 114
FRONTIER_PROGRAMS = ("candidate_b128", "candidate_b114", "control_b128", "control_b114")
PROGRAMS = ("wk_decode", "wk_promote", *FRONTIER_PROGRAMS,
            "exact_decode", "exact_promote", "observer")
BUCKET = "driftbench-dsv4-uc"
RESERVE = 1 << 30
ORIGINALS_LIMIT = 128 << 20
# Refusal caps for actual compiler reports, not measured allocation claims.
# The four-producer materializer has 106,741,760B decoded / 213,696,512B
# promoted abstract outputs per chip. Its two jobs need separate output caps.
def memory_caps(program: str) -> dict[str, int]:
    if program not in PROGRAMS:
        raise ValueError("history unregistered compiler memory program")
    return dict(
        argument_size_in_bytes=2 << 30,
        output_size_in_bytes={"exact_decode": 128 << 20, "exact_promote": 256 << 20}.get(program, 96 << 20),
        alias_size_in_bytes=0, temp_size_in_bytes=1 << 30,
        generated_code_size_in_bytes=128 << 20,
    )


# Retained originals: the receipts are committed under docs/artifacts and bound
# here by content digest; each names every rank's original runner objects by
# cloud generation/size/CRC/SHA. Originals are keyed by LAUNCH rank (host).
ORIGINAL_TAGS = dict(
    candidate="greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T225859315457683Z",
    control="greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_live32_20260909T103047459508942Z",
)
RECEIPTS = dict(
    candidate=("docs/artifacts/prefill-canonical8k-token-refusal-20260909.json",
               "f659925dec463a1473076d913cfd577a81ba0d584c9feba30acb71f5480df175"),
    control=("docs/artifacts/prefill-frozen-live32-diagnostic-20260909.json",
             "1c597126721ccea76c58e5ce066be82a38fc0975688dd37037c6b9c7793571f0"),
)
ORIGINAL_FORMS = ("json", "npz")
MAX_ORIGINAL_BYTES = 64 << 20


class Step(NamedTuple):
    index: int
    group: int
    branch: str
    program: str
    offset: int
    count: int


def is_tag(tag: str) -> bool:
    return isinstance(tag, str) and re.fullmatch(
        r"greenfield_fp8_ws32_history_frontier_l06_[0-9]{8}T[0-9]+Z", tag) is not None


def _blocks(prompt_length: int, block: int) -> list[tuple[int, int]]:
    """Contiguous (offset, count) blocks covering the prompt, tail 1..block."""
    full = (prompt_length - 1) // block
    tail = prompt_length - full * block
    if not 1 <= tail <= block or full * block + tail != prompt_length:
        raise AssertionError("history block plan drifted")
    return [(k * block, block) for k in range(full)] + [(full * block, tail)]


def plan(prompt_length: int = PROMPT_LENGTH, *, wide: int = WIDE, narrow: int = NARROW) -> tuple[Step, ...]:
    """Interleave one wide candidate block with the narrow control blocks it spans.

    Both branches cover [0, prompt_length) contiguously with monotonic offsets.
    A tail runs in the physical-114 program when it fits, exactly as the
    production 63x128+91 and 254x32+27 schedules do; a wider tail keeps the
    physical-128 program. The interleaving bounds the host rows compared at
    once to one wide block per group.
    """
    for name, value in (("prompt_length", prompt_length), ("wide", wide), ("narrow", narrow)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"history plan {name} must be a positive integer")
    if wide % narrow or wide > WIDE or narrow > wide:
        raise ValueError("history plan requires narrow | wide <= 128")

    def program(branch: str, count: int, rows: int) -> str:
        if count == rows or count > TAIL_ROWS:
            return f"{branch}_b128"
        return f"{branch}_b114"

    candidate = _blocks(prompt_length, wide)
    control = iter(_blocks(prompt_length, narrow))
    pending = next(control, None)
    steps: list[Step] = []
    for group, (offset, count) in enumerate(candidate):
        steps.append(Step(len(steps), group, "candidate", program("candidate", count, wide), offset, count))
        while pending is not None and pending[0] < offset + count:
            steps.append(Step(len(steps), group, "control", program("control", pending[1], narrow), *pending))
            pending = next(control, None)
    if pending is not None:
        raise AssertionError("history control blocks outlived the candidate plan")
    for branch in BRANCHES:
        own = [s for s in steps if s.branch == branch]
        covered = 0
        for step in own:
            if step.offset != covered or step.count <= 0:
                raise AssertionError("history plan is not contiguous")
            covered += step.count
        if covered != prompt_length:
            raise AssertionError("history plan does not cover the prompt")
    return tuple(steps)


def receipt(repo: Path, branch: str) -> dict[str, Any]:
    """The committed receipt for one branch, bound by its content digest."""
    relative, digest = RECEIPTS[branch]
    raw = (Path(repo) / relative).read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f"history {branch} receipt digest differs")
    value = json.loads(raw)
    if value.get("tag") != ORIGINAL_TAGS[branch]:
        raise ValueError(f"history {branch} receipt names another run")
    return value


def _pin(entry: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    generation = str(entry["generation"])
    if (not generation.isdecimal() or int(generation) <= 0
            or type(entry["size"]) is not int or not 0 < entry["size"] <= MAX_ORIGINAL_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or not isinstance(entry["crc32c"], str) or not entry["crc32c"]):
        raise ValueError("history original pin invalid")
    return dict(name=name, generation=generation, size=entry["size"],
                crc32c=entry["crc32c"], sha256=entry["sha256"])


def original_pins(repo: Path, rank: int) -> dict[str, dict[str, dict[str, Any]]]:
    """Cloud object pins of this launch rank's original runner json/npz per branch."""
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("history originals require launcher rank0..7")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    candidate = receipt(repo, "candidate")
    ranks = [r for r in candidate["rank_records"] if r.get("rank") == rank]
    if len(ranks) != 1:
        raise ValueError("history candidate receipt lacks this rank")
    prefix = f"results/{ORIGINAL_TAGS['candidate']}/host_records/runner.rank{rank}."
    by_form = {}
    for entry in ranks[0]["records"]:
        if entry["name"].startswith(prefix):
            by_form[entry["name"][len(prefix):]] = _pin(entry, name=entry["name"])
    result["candidate"] = {form: by_form[form] for form in ORIGINAL_FORMS}
    control = receipt(repo, "control")
    hosts = [h for h in control["hosts"] if h.get("rank") == rank]
    if len(hosts) != 1:
        raise ValueError("history control receipt lacks this rank")
    originals = hosts[0]["originals"]
    expected_uri = f"gs://{BUCKET}/results/{ORIGINAL_TAGS['control']}/host_records/runner.rank{rank}."
    control_pins = {}
    for form in ORIGINAL_FORMS:
        entry = originals[form]
        if entry["uri"] != expected_uri + form:
            raise ValueError("history control original uri differs")
        control_pins[form] = _pin(entry, name=entry["uri"][len(f"gs://{BUCKET}/"):])
    if control_pins["npz"]["sha256"] != hosts[0]["npz_sha256"]:
        raise ValueError("history control receipt npz digest disagrees with itself")
    result["control"] = control_pins
    return result


def reproduction_rows(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Step0 event0..3 arrays of one original runner NPZ, for producers0/1/2/6."""
    producers = np.asarray(arrays["dsa_producer_layer_ids"])
    if producers.ndim != 1 or producers[: len(PRODUCERS)].tolist() != list(PRODUCERS):
        raise ValueError("history original producers differ")
    rows = {}
    for key, name in (("dsa_selected_positions", "positions"),
                      ("dsa_selected_valid_counts", "counts"),
                      ("dsa_selected_scores", "scores")):
        value = np.asarray(arrays[key])
        if value.ndim < 3 or value.shape[1] < len(PRODUCERS) or value.shape[2] != 1:
            raise ValueError("history original observation geometry differs")
        rows[name] = np.ascontiguousarray(value[0, : len(PRODUCERS)]).copy()
    if (rows["positions"].shape != (len(PRODUCERS), 1, rows["positions"].shape[-1])
            or rows["counts"].shape != (len(PRODUCERS), 1)
            or rows["scores"].shape != rows["positions"].shape
            or rows["positions"].dtype != np.int32 or rows["counts"].dtype != np.int32
            or rows["scores"].dtype != np.float32):
        raise ValueError("history original observation dtype/shape differs")
    return rows


__all__ = ["BRANCHES", "CAPACITY", "FRONTIER_PROGRAMS", "KERNEL", "LAYERS",
           "ORIGINALS_LIMIT", "ORIGINAL_TAGS", "PAYLOAD_BYTES", "PRODUCERS", "PROGRAMS",
           "PROMPT_LENGTH", "PROMPT_SHA", "PROTOCOL", "RESERVE", "Step", "WITNESS",
           "is_tag", "memory_caps", "original_pins", "plan", "receipt", "reproduction_rows"]
