"""Narrow debug-coordinate equivalence, not general HLO normalization.

Baseline allows only existing main/<module> locations of the protected worker.
The paired-only function also allows coordinates in three reviewed source files
to move: DSA plus the WS32 and router files with default-off observation hooks.
All executable text, other model locations, IDs and other debug tables remain bytes.
Raw hashes must always be retained independently of this comparison digest.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any


WORKER = (
    "/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/run_short_decoder_ws32.py"
)
DSA = "/home/gianl/glm-tpu-topology-rewrite/glm_tpu/greenfield/kernels/reference/dsa.py"
WS32 = "/home/gianl/glm-tpu-topology-rewrite/glm_tpu/greenfield/kernels/ws32.py"
MOE = "/home/gianl/glm-tpu-topology-rewrite/glm_tpu/greenfield/kernels/reference/moe.py"


def paired_worker_dsa_location_identity(text: str) -> dict[str, Any]:
    """Paired-only relocation of reviewed source, retaining every other byte.

    The caller binds the canonical digest to an original graph. This does not
    authorize changed functions, callsites, stack structure or executable text.
    """
    worker = worker_location_identity(text)
    files = re.search(r"(?ms)^FileNames\n(.*?)\n\nFunctionNames\n", text)
    locations = re.search(r"(?ms)^FileLocations\n(.*?)\n\nStackFrames\n", text)
    assert files is not None and locations is not None  # validated above
    source_ids = {}
    for label, filename in (("dsa", DSA), ("ws32", WS32), ("moe", MOE)):
        ids = re.findall(r'(?m)^(\d+) "' + re.escape(filename) + r'"$', files[1])
        if len(ids) > 1:
            raise ValueError(f"ambiguous paired {label} filename")
        if ids:
            source_ids[ids[0]] = label
    worker_ids = {entry["location_id"] for entry in worker["worker_locations"]}
    pattern = re.compile(
        r"(?m)^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) "
        r"line=(\d+) end_line=(\d+) column=(\d+) end_column=(\d+)\}$"
    )
    source_locations = {label: [] for label in ("dsa", "ws32", "moe")}

    def replace(match: re.Match) -> str:
        source = source_ids.get(match[2])
        if source is not None:
            source_locations[source].append(
                dict(
                    location_id=int(match[1]),
                    file_name_id=int(match[2]),
                    function_name_id=int(match[3]),
                    line=int(match[4]),
                    end_line=int(match[5]),
                    column=int(match[6]),
                    end_column=int(match[7]),
                )
            )
        if source is not None or int(match[1]) in worker_ids:
            return (
                f"{match[1]} {{file_name_id={match[2]} function_name_id={match[3]} "
                "line=0 end_line=0 column=0 end_column=0}"
            )
        return match[0]

    canonical_locations = pattern.sub(replace, locations[1])
    for label in source_ids.values():
        if not source_locations[label]:
            raise ValueError(f"paired {label} filename has no supported locations")
    canonical = (
        text[: locations.start(1)] + canonical_locations + text[locations.end(1) :]
    )
    return dict(
        schema_version="ws32_paired_reviewed_source_debug_coordinates_v2",
        raw_optimized_hlo_sha256=worker["raw_optimized_hlo_sha256"],
        paired_location_equivalence_sha256=sha256(canonical.encode()).hexdigest(),
        worker_locations=worker["worker_locations"],
        **{
            f"{label}_locations": entries for label, entries in source_locations.items()
        },
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
