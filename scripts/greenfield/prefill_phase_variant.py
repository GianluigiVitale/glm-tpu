"""Explicit, default-off phase candidates; historical DB595 remains unchanged."""

from dataclasses import dataclass
import re
from typing import Any, Mapping

PAIRED_KERNEL = "ws32_prefill_paired_sort_phase"
PAIRED_PROTOCOL = "ws32-layer6-db594-paired-position-sort-phase-v1"


@dataclass(frozen=True)
class Variant:
    kernel: str
    protocol: str
    admission: Any
    budgeter: Any
    paired_position_sort: bool


def variants() -> tuple[Variant, ...]:
    from scripts.greenfield import prefill_phase_baseline as baseline
    from scripts.greenfield import prefill_completed_window_admission as original
    from scripts.greenfield import prefill_completed_window_assembly as assembly
    from scripts.greenfield import prefill_paired_sort_admission as paired

    return (
        Variant(
            baseline.KERNEL, baseline.PROTOCOL, original, assembly.memory_budget, False
        ),
        Variant(PAIRED_KERNEL, PAIRED_PROTOCOL, paired, paired.memory_budget, True),
    )


def for_tag(tag: str) -> Variant:
    for value in variants():
        if re.fullmatch(r"greenfield_fp8_" + value.kernel + r"_l6_[a-zA-Z0-9_]+", tag):
            return value
    raise ValueError("unknown prefill phase tag")


def for_record(record: Mapping[str, Any]) -> Variant:
    for value in variants():
        if record.get("protocol") == value.protocol:
            if record.get("profile") != value.admission.PROFILE:
                raise ValueError("mixed phase protocol/profile")
            if "kernel" in record and record["kernel"] != value.kernel:
                raise ValueError("mixed phase protocol/kernel")
            return value
    raise ValueError("unknown prefill phase protocol")
