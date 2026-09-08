"""Narrow debug-coordinate equivalence, not general HLO normalization.

Only the existing main/<module> FileLocations of the protected worker may move.
All executable text, model locations, IDs and other debug tables remain bytes.
Raw hashes must always be retained independently of this comparison digest.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any


WORKER = (
    "/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_short_decoder_ws32.py"
)


def worker_location_identity(text: str) -> dict[str, Any]:
    def table(name: str, following: str) -> re.Match:
        matches = list(
            re.finditer(r"(?ms)^" + name + r"\n(.*?)\n\n" + following + r"\n", text)
        )
        if len(matches) != 1:
            raise ValueError("ambiguous/missing HLO debug table")
        return matches[0]

    files = table("FileNames", "FunctionNames")
    worker_ids = re.findall(r'(?m)^(\d+) "' + re.escape(WORKER) + r'"$', files[1])
    if len(worker_ids) != 1:
        raise ValueError("HLO lacks unique exact worker filename")
    functions = table("FunctionNames", "FileLocations")
    names = dict(re.findall(r'(?m)^(\d+) "([^"\n]*)"$', functions[1]))
    locations = table("FileLocations", "StackFrames")
    pattern = re.compile(
        r"(?m)^(\d+) \{file_name_id="
        + worker_ids[0]
        + r" function_name_id=(\d+) line=(\d+) end_line=(\d+) column=(\d+) end_column=(\d+)\}$"
    )
    entries = list(pattern.finditer(locations[1]))
    if len(entries) != 2 or {names.get(m[2]) for m in entries} != {"main", "<module>"}:
        raise ValueError("unacquired worker debug callsite structure")
    actual = [
        dict(
            location_id=int(m[1]),
            function=names[m[2]],
            line=int(m[3]),
            end_line=int(m[4]),
            column=int(m[5]),
            end_column=int(m[6]),
        )
        for m in entries
    ]

    def replace(match: re.Match) -> str:
        return (
            f"{match[1]} {{file_name_id={worker_ids[0]} function_name_id={match[2]} "
            "line=0 end_line=0 column=0 end_column=0}"
        )

    canonical_locations = pattern.sub(replace, locations[1])
    # Splice just the table payload. Everything outside these eight integer
    # coordinates, including instruction metadata and backend_config, stays.
    canonical = (
        text[: locations.start(1)] + canonical_locations + text[locations.end(1) :]
    )
    return dict(
        schema_version="ws32_worker_debug_coordinates_v1",
        raw_optimized_hlo_sha256=sha256(text.encode()).hexdigest(),
        worker_location_equivalence_sha256=sha256(canonical.encode()).hexdigest(),
        worker_locations=actual,
    )
