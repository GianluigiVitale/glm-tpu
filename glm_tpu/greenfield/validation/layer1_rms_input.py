"""Tombstone the rejected callback-based layer-1 RMS-input observer.

The protected 2026-08-29 run reproduced DB551's observer-perturbation
fingerprint exactly. A ``jax.debug.callback`` is a consuming operation and can
split the fused producer/add/RMSNorm schedule even when the hook returns
nothing. The captured BF16 operands and host-side FP32 sum are therefore
diagnostic bytes, never an accepted boundary oracle.

The former config and callable entrypoints remain so stale callers fail with
the evidence-backed reason instead of an import error or silently recreating
an ``ACCEPTED_*`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

SOURCE_KIND = "glm52_legacy_layer1_rms_input_operands"
REJECTED_KIND = "glm52_rejected_layer1_rms_input_observer_perturbation"
REJECTION_CLASSIFICATION = "REJECTED_OBSERVER_PERTURBATION"
REJECTED_LEGACY_PIN = "8dc7d20fedca5a98c27bfd1774827305973fa4c1"
REJECTED_RUN_TAG = "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z"
REJECTION_MESSAGE = (
    f"{REJECTION_CLASSIFICATION}: layer-1 RMS-input callback observer is "
    "tombstoned; protected run "
    f"{REJECTED_RUN_TAG} reproduced DB551 observer perturbation (557434 "
    "selected-position and 573438 selected-score mismatches, first at "
    "event 1); callback-derived bytes cannot be sealed as accepted evidence"
)


@dataclass(frozen=True, slots=True)
class Layer1RmsInputCaptureConfig:
    """Compatibility shape for stale callers; every operation is refused."""

    source_dump_dir: Path
    output_dir: Path
    db550_boundary_path: Path
    straddler_classification_path: Path
    vllm_repository: Path
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_db550_sha256: str = ""
    expected_straddler_sha256: str = ""
    expected_vllm_pin: str = ""
    expected_vllm_ir_layernorm_sha256: str = ""
    expected_vllm_executor_layernorm_sha256: str = ""
    expected_model_id: str = "zai-org/GLM-5.2-FP8"
    expected_layer_name: str = "model.layers.1.input_layernorm"
    expected_position: int = 8155
    expected_process_count: int = 8
    expected_capture_process_index: int = 0
    expected_source_row: int = 0


def _refuse_rejected_observer() -> NoReturn:
    raise RuntimeError(REJECTION_MESSAGE)


def capture_accepted_layer1_rms_input(
    config: Layer1RmsInputCaptureConfig,
) -> NoReturn:
    """Refuse the rejected observer before reading inputs or creating output."""

    del config
    _refuse_rejected_observer()


def validate_layer1_rms_input_artifacts(
    config: Layer1RmsInputCaptureConfig,
) -> NoReturn:
    """Refuse promotion/revalidation of callback-derived artifacts."""

    del config
    _refuse_rejected_observer()
