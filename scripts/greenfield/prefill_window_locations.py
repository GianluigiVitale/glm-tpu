"""DB590's seven exact host coordinates; every other optimized byte is retained.

Adapted from validation/ws32_hlo_worker_locations.py. No frame contraction,
function/ID remapping, model-location masking or general HLO normalization.
"""

from hashlib import sha256
import re
from typing import Any

ROOT = "/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/"
HOST_LOCATIONS = (
    (ROOT + "probe_ws32_prefill_layer.py", "<module>"),
    (ROOT + "probe_ws32_prefill_layer.py", "main"),
    (ROOT + "prefill_window_acquisition.py", "execute_acquisition"),
    (ROOT + "prefill_window_acquisition.py", "acquire_programs"),
    (ROOT + "prefill_window_acquisition.py", "fleet_step"),
    (ROOT + "prefill_window_acquisition.py", "acquire_programs.<locals>.<lambda>"),
    (ROOT + "probe_ws32_prefill_layer.py", "compile_program"),
)


def location_identity(text: str) -> dict[str, Any]:
    def table(name: str, following: str) -> re.Match:
        matches = list(
            re.finditer(r"(?ms)^" + name + r"\n(.*?)\n\n" + following + r"\n", text)
        )
        if len(matches) != 1:
            raise ValueError("window missing/ambiguous debug table")
        return matches[0]

    files = dict(
        re.findall(r'(?m)^(\d+) "([^"\n]*)"$', table("FileNames", "FunctionNames")[1])
    )
    functions = dict(
        re.findall(
            r'(?m)^(\d+) "([^"\n]*)"$', table("FunctionNames", "FileLocations")[1]
        )
    )
    locations = table("FileLocations", "StackFrames")
    pattern = re.compile(
        r"(?m)^(\d+) \{file_name_id=(\d+) function_name_id=(\d+) line=(\d+) end_line=(\d+) column=(\d+) end_column=(\d+)\}$"
    )
    selected, actual = {}, []
    for match in pattern.finditer(locations[1]):
        key = (files.get(match[2]), functions.get(match[3]))
        if key not in HOST_LOCATIONS:
            continue
        if key in selected:
            raise ValueError("window duplicate host coordinate location")
        selected[key] = match[1]
        actual.append(
            dict(
                location_id=int(match[1]),
                filename=key[0],
                function=key[1],
                line=int(match[4]),
                end_line=int(match[5]),
                column=int(match[6]),
                end_column=int(match[7]),
            )
        )
    if set(selected) != set(HOST_LOCATIONS):
        raise ValueError("window changed outer compilation stack")

    def replace(match: re.Match) -> str:
        key = (files.get(match[2]), functions.get(match[3]))
        if key not in selected:
            return match[0]
        return (
            f"{match[1]} {{file_name_id={match[2]} function_name_id={match[3]} "
            "line=0 end_line=0 column=0 end_column=0}"
        )

    canonical = (
        text[: locations.start(1)]
        + pattern.sub(replace, locations[1])
        + text[locations.end(1) :]
    )
    return dict(
        schema_version="ws32_window_host_coordinates_v1",
        raw_optimized_hlo_sha256=sha256(text.encode()).hexdigest(),
        host_location_equivalence_sha256=sha256(canonical.encode()).hexdigest(),
        host_locations=actual,
    )
