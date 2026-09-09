"""Panel B128 bounded output against byte-exact DB594 B32 controls.

Only wide/actual output may differ. All other first witnesses remain original;
later wide/actual outputs must repeat their own first bytes. The collector
independently reconstructs assemblies and replays the unchanged suffix bounds.
"""

from pathlib import Path
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_completed_window_protocol as completed

CHANGED = frozenset(("wide", "actual"))


def check_observation(
    original: Mapping[str, Any],
    *,
    slot: int,
    kind: str,
    values: Mapping[str, np.ndarray]
) -> dict:
    if kind not in CHANGED:
        return originals.check_observation(
            original, slot=slot, kind=kind, values=values
        )
    if type(slot) is not int or not 0 <= slot < 32:
        raise ValueError("panel original slot differs")
    expected = original["owners"][str(slot)]["components"][kind]
    observed = originals.manifest(values)
    if set(observed) != set(expected) or any(
        observed[k] != expected[k] for k in expected if k != "output"
    ):
        raise ValueError("panel non-output original witness differs")
    value = np.asarray(values["output"])
    if value.dtype == np.uint16:
        value = value.view(completed.window.BF16)
    if (
        value.dtype != completed.window.BF16
        or list(value.shape) != expected["output"]["shape"]
        or not np.isfinite(value).all()
    ):
        raise ValueError("panel output geometry/dtype/finiteness differs")
    return observed


def bounded_replay(path: Path, slots: Mapping[int, int]) -> dict:
    result = completed.replay_case(path, case="competitive", slots_by_device=slots)
    if not result["passed"]:
        raise ValueError("panel B128 output fails original B32 suffix bounds")
    return result


def bounded_receipt(path: Path, slots: Mapping[int, int]) -> dict:
    """Replay fully, but do not rewrite a half-MB report at every phase.

    The original NPZ and pinned replay code retain every detail. The collector
    recomputes this digest from those arrays, never trusts the producer verdict.
    """
    result = bounded_replay(path, slots)
    raw = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return dict(
        format="panel_bounded_replay_sha256_v1",
        passed=True,
        report_sha256=sha256(raw).hexdigest(),
        report_bytes=len(raw),
    )


class PanelOriginalVerifier(originals.OriginalVerifier):
    def __init__(
        self, calls: Any, original: Mapping[str, Any], inputs: Mapping[str, np.ndarray]
    ) -> None:
        super().__init__(calls, original, inputs)
        self.first_candidate: dict[tuple[int, str], dict] = {}

    def check(self, *, slot: int, kind: str, values: Mapping[str, np.ndarray]) -> None:
        observed = check_observation(self.original, slot=slot, kind=kind, values=values)
        if kind in CHANGED:
            key = slot, kind
            if key not in self.first_candidate:
                self.first_candidate[key] = observed
            elif self.first_candidate[key] != observed:
                raise ValueError("panel candidate does not repeat first output bytes")

    def capture(
        self, kind: str, observed: Mapping[int, Mapping[str, np.ndarray]]
    ) -> None:
        super().capture(kind, observed)
        if kind == "control" and self.visits[kind] == 1:
            # The first full traversal is WARMUP. This voted preservation hook
            # must pass before another traversal (and any timed sample) starts.
            self.report["panel_bounded_reference"] = bounded_receipt(
                self.calls.root / "phase_first.npz", self.calls.local_slots
            )

    def finish(self, traversals: int) -> None:
        if "panel_bounded_reference" not in self.report:
            raise ValueError("panel run lacks first bounded original comparison")
        super().finish(traversals)
