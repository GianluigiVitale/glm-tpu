"""Safety verdicts of the program child (G1/G2 frozen ``safety``; G1-protocol/G2-protocol ``verdicts``).

Two kinds of record come from here:

* ``admission_cases()``: production's memory admission (``project_memory``, recorded as
  ``memory_projection``) and HLO admission (``check_hlo_collectives``, recorded as
  ``inspect_research_hlo``) run on fixed synthetic inputs. The memory cases sit on the 512 MiB
  reserve boundary (one byte below and exactly at the chip limit), exercise the alias credit, the
  per-chip rule and the refusals of invalid accounting; the HLO cases are a small synthetic
  optimized module that follows the physical expert-8/feature-4 axes (accepted, with a full-pod
  all-reduce of exactly 4 KiB) and four variants that must be refused (a full-pod payload one
  element over 4 KiB, an all-to-all, non-physical replica groups, no collectives). Both functions
  are found under their 181c013e names, the current names of the permanent table
  ``driver.RECORDED_NAMES`` or the current names ``closure_map.toml`` ``[functions]`` maps to them
  (``driver.current_names``); a missing one is recorded (``<absent>``), never raised. A rename
  changes its ``RECORDED_NAMES`` row in the same commit: the frozen record cannot be re-recorded.
* ``run_safety(protocol)``: the per-run facts that no refactor may change, taken from the load
  protocol the real ``__init__``/``_load``/``compile`` followed: the graph-consensus probe verdict,
  the arguments of the checkpoint verification call (``verify_file_hashes=True``,
  ``local_slot_layout=True``, the four local slots, the site pins), the memory admission requests
  (order-insensitive), the programs that pass the HLO admission and the number of graph-consensus
  calls.

The frozen record (``safety_verdicts`` and ``run_safety``) keeps verdicts only (accepted / refused
and the exception type, fits or not), so message wording and report contents -- which the work
units change (WU-R renamed the admission functions and reworded the HLO refusals; H11 renames the HLO
admission profile string) -- stay in the re-baselined characterization record; the frozen G1/G2 files
keep what must never change.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

GIB = 1 << 30
LIMIT = 32 * GIB
IN_USE = 30 * GIB
RESERVE = 512 * (1 << 20)  # the 181c013e reserve the boundary cases are placed around
_FIELDS = ("output_size_in_bytes", "temp_size_in_bytes", "generated_code_size_in_bytes", "alias_size_in_bytes")


def _memory(output: Any, temp: Any = 0, code: Any = 0, alias: Any = 0) -> dict[str, Any]:
    return dict(zip(_FIELDS, (output, temp, code, alias), strict=True))


def _chips(*in_use: int, limit: int = LIMIT, ids: tuple[int, ...] = (0, 1, 2, 3)) -> list[dict[str, Any]]:
    return [
        dict(device_id=device, bytes_in_use=used, bytes_limit=limit, peak_bytes_in_use=used)
        for device, used in zip(ids, in_use, strict=True)
    ]


FITS = LIMIT - IN_USE - RESERVE - 1  # predicted = limit - 1
MEMORY_CASES: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    "one_byte_below_limit": (_chips(*(IN_USE,) * 4), _memory(FITS - 3, 1, 2)),
    "at_limit": (_chips(*(IN_USE,) * 4), _memory(FITS + 1)),
    "alias_credit": (_chips(*(IN_USE,) * 4), _memory(FITS + GIB, alias=GIB)),
    "alias_exceeds_outputs": (_chips(*(IN_USE,) * 4), _memory(0, alias=GIB)),
    "one_chip_fuller": (_chips(IN_USE, IN_USE, IN_USE + 1, IN_USE), _memory(FITS)),
    "negative_field": (_chips(*(IN_USE,) * 4), _memory(-1)),
    "float_field": (_chips(*(IN_USE,) * 4), _memory(1.0)),
    "three_chips": (_chips(*(IN_USE,) * 3, ids=(0, 1, 2)), _memory(1)),
    "duplicate_chip": (_chips(*(IN_USE,) * 4, ids=(0, 1, 2, 2)), _memory(1)),
    "in_use_over_limit": (_chips(IN_USE, IN_USE, LIMIT + 1, IN_USE), _memory(1)),
}

_FEATURE = "{" + ",".join("{" + ",".join(str(i) for i in range(r * 4, r * 4 + 4)) + "}" for r in range(8)) + "}"
_EXPERT = "{" + ",".join("{" + ",".join(str(i) for i in range(c, 32, 4)) + "}" for c in range(4)) + "}"
_POD = "{{" + ",".join(str(i) for i in range(32)) + "}}"
_PAIRS = "{" + ",".join("{" + f"{i},{i + 1}" + "}" for i in range(0, 32, 2)) + "}"


def synthetic_hlo(
    *, pod_elements: int = 1024, extra: str = "", feature_groups: str = _FEATURE, collectives: bool = True
) -> str:
    """A small optimized-HLO module in the form the TPU compiler prints: 32 partitions, an
    all-reduce over the feature axis, an all-gather over the expert axis and a full-pod
    all-reduce of ``pod_elements`` f32 values (1,024 = the 4 KiB full-pod limit)."""
    tile, pod = "f32[8,128]{1,0}", f"f32[{pod_elements}]{{0}}"
    reduce, gather = "use_global_device_ids=true, to_apply=%add", "dimensions={0}, use_global_device_ids=true"
    if collectives:
        body = (
            f"  %ar = {tile} all-reduce({tile} %c), channel_id=1, replica_groups={feature_groups}, {reduce}\n"
            f"  %ag = f32[64,128]{{1,0}} all-gather({tile} %ar), channel_id=2, replica_groups={_EXPERT}, {gather}\n"
            f"  %s = {pod} slice(f32[64,128]{{1,0}} %ag), slice={{[0:{pod_elements}]}}\n"
            f"  %pr = {pod} all-reduce({pod} %s), channel_id=3, replica_groups={_POD}, {reduce}\n"
        )
    else:
        body = f"  %ar = {tile} negate({tile} %c)\n"
    return (
        "HloModule glm_equivalence_synthetic, entry_computation_layout={(bf16[8,128]{1,0})->bf16[8,128]{1,0}}, "
        "num_partitions=32\n\n"
        "%add (x: f32[], y: f32[]) -> f32[] {\n  %x = f32[] parameter(0)\n  %y = f32[] parameter(1)\n"
        "  ROOT %sum = f32[] add(f32[] %x, f32[] %y)\n}\n\n"
        "ENTRY %main (p0: bf16[8,128]) -> bf16[8,128] {\n  %p0 = bf16[8,128]{1,0} parameter(0)\n"
        f"  %c = {tile} convert(bf16[8,128]{{1,0}} %p0)\n{body}{extra}"
        f"  ROOT %out = bf16[8,128]{{1,0}} convert({tile} %ar)\n}}\n"
    )


HLO_CASES: dict[str, str] = {
    "physical_axes_pod_4096_bytes": synthetic_hlo(),
    "pod_payload_4100_bytes": synthetic_hlo(pod_elements=1025),
    "all_to_all": synthetic_hlo(
        extra="  %a2a = f32[8,128]{1,0} all-to-all(f32[8,128]{1,0} %ar), channel_id=4, "
        f"replica_groups={_FEATURE}, dimensions={{0}}, use_global_device_ids=true\n"
    ),
    "non_physical_groups": synthetic_hlo(feature_groups=_PAIRS),
    "no_collectives": synthetic_hlo(collectives=False),
}


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    """(verdict, detail): ``accepted`` with the returned report, or ``refused (<type>)`` with the message."""
    try:
        report = call()
    except Exception as exc:  # the refusal is the recorded result
        return f"refused ({type(exc).__name__})", f"{type(exc).__name__}: {exc}"
    return "accepted", report


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=lambda item: f"<{type(item).__name__}>"))


def admission_cases() -> dict[str, Any]:
    """Full records (characterization): ``{memory: {case: {verdict, detail}}, hlo: {...}}``."""
    from .driver import _definition

    out: dict[str, Any] = {}
    project_memory = _definition("memory_projection")  # recorded (181c013e) names: driver.RECORDED_NAMES
    check_hlo_collectives = _definition("inspect_research_hlo")
    for label, function, cases in (
        ("memory", project_memory, MEMORY_CASES),
        ("hlo", check_hlo_collectives, HLO_CASES),
    ):
        if isinstance(function, str):
            out[label] = function  # <absent> or <ambiguous: ...>
            continue
        rows = {}
        for case, value in cases.items():
            if label == "memory":
                stats, memory = value
                verdict, detail = _outcome(lambda s=stats, m=memory: function(s, m))  # noqa: B023 (called in this iteration)
                if verdict == "accepted":
                    verdict = "fits" if detail.get("passed") is True else "does not fit"
            else:
                verdict, detail = _outcome(lambda text=value: function(text))  # noqa: B023 (called in this iteration)
            rows[case] = dict(verdict=verdict, detail=_json_safe(detail))
        out[label] = rows
    return out


def safety_verdicts(cases: dict[str, Any]) -> dict[str, Any]:
    """The frozen part of ``admission_cases()``: verdicts only."""
    return {
        label: rows if isinstance(rows, str) else {case: row["verdict"] for case, row in rows.items()}
        for label, rows in cases.items()
    }


def _probe_verdict(outcome: str | None) -> str:
    if outcome is None:
        return "<absent>"  # the probe did not run (a changed driver); the frozen record then differs
    if outcome.startswith("accepted"):
        return "accepted"
    return f"refused ({outcome.split(':', 1)[0]})"


def run_safety(protocol: dict[str, Any]) -> dict[str, Any]:
    """Frozen per-run facts from one run's recorded load protocol (see the module docstring)."""
    admissions = sorted(protocol.get("admissions", []), key=lambda row: json.dumps(row, sort_keys=True))
    compile_record = protocol.get("compile", {})
    return dict(
        graph_consensus_probe=_probe_verdict(protocol.get("probes", {}).get("graph_consensus")),
        verify_checkpoint=protocol.get("loader", {}).get("verify_ws32_runtime_checkpoint", "<absent>"),
        memory_admission_requests=admissions,
        hlo_admitted_programs=sorted(compile_record.get("hlo_admissions", [])),
        graph_consensus_calls=compile_record.get("consensus_calls"),
    )
