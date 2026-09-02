from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.validation.gate_d_projection_contraction_numerical import (
    EXPECTED_CURRENT_KEY_OWNER_SHA256,
    EXPECTED_NORMALIZED_OWNER_SHA256,
    EXPECTED_PROJECTED_KEY_OWNER_SHA256,
    classify_projection_outputs,
    projection_input_records,
    projection_state_records,
    validate_predecessors,
    validate_wk_weight,
)

ROOT = Path(__file__).parents[3]
CAPSULE = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _predecessors() -> dict[str, dict]:
    return {
        "capsule": _json(CAPSULE / "capsule.json"),
        "execution_authority": _json(Path(f"{CAPSULE}-execution-authority.json")),
        "host_materialization": _json(
            ROOT
            / "docs/artifacts/gate-d-pp16-numerical-host-materialization-equivalence.json"
        ),
        "frontier": _json(
            ROOT / "docs/artifacts/gate-d-projection-arithmetic-frontier-analysis.json"
        ),
        "hlo_success": _json(
            ROOT
            / "docs/artifacts/gate-d-projection-contraction-pp16-success-hlo-adjudication.json"
        ),
    }


def test_predecessors_and_exact_operand_catalogues_pass() -> None:
    values = _predecessors()
    result = validate_predecessors(**values)
    assert set(result) == {
        "capsule_input_sha256",
        "capsule_sha256",
        "capsule_state_sha256",
        "execution_authority_sha256",
        "frontier_authority_sha256",
        "hlo_success_authority_sha256",
        "host_materialization_authority_sha256",
    }
    assert set(projection_input_records(values["execution_authority"])) == {
        "key_norm_bias_bf16_bits",
        "key_norm_weight_bf16_bits",
        "wk_scale_inv",
        "wk_weight_bits",
    }
    assert set(projection_state_records(values["capsule"])) == {"normalized"}


@pytest.mark.parametrize(
    ("section", "path", "replacement"),
    [
        ("capsule", ("artifact", "sha256"), "0" * 64),
        ("execution_authority", ("authority_kind",), "wrong"),
        ("host_materialization", ("comparison", "wk_weight_fp32_sha256"), "0" * 64),
        ("frontier", ("root_cause_proven",), True),
        ("frontier", ("cpu_jax_dot_control", "projection_sha256"), "0" * 64),
        ("hlo_success", ("authorization", "numerical_execution"), True),
        ("hlo_success", ("optimized_hlo", "sha256"), "0" * 64),
    ],
)
def test_predecessors_reject_contradictions(
    section: str, path: tuple[str, ...], replacement: object
) -> None:
    values = _predecessors()
    target = values[section]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(RuntimeError, match="drifted"):
        validate_predecessors(**values)


def test_normalized_watchpoint_rejects_one_owner_change() -> None:
    values = _predecessors()
    capsule = copy.deepcopy(values["capsule"])
    normalized = next(
        item for item in capsule["watchpoints"] if item["id"] == "layer1.normalized"
    )
    normalized["arrays"][1]["array_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="normalized watchpoint"):
        projection_state_records(capsule)


def test_classifier_accepts_exact_synthetic_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized = np.zeros((2, 1, 6144), dtype=np.uint16)
    projected = np.zeros((2, 1, 128), dtype=np.float32)
    current = np.zeros((2, 1, 128), dtype=np.float32)
    expected = iter(
        [
            EXPECTED_NORMALIZED_OWNER_SHA256,
            EXPECTED_NORMALIZED_OWNER_SHA256,
            EXPECTED_PROJECTED_KEY_OWNER_SHA256,
            EXPECTED_PROJECTED_KEY_OWNER_SHA256,
            EXPECTED_CURRENT_KEY_OWNER_SHA256,
            EXPECTED_CURRENT_KEY_OWNER_SHA256,
        ]
    )
    module = importlib.import_module(
        "glm_tpu.greenfield.validation.gate_d_projection_contraction_numerical"
    )
    monkeypatch.setattr(module, "_array_sha256", lambda _value, _np: next(expected))
    # NumPy has no native BF16 in this test environment; the classifier's dtype
    # guard is exercised separately on the sealed runtime.
    monkeypatch.setattr(
        module,
        "str",
        lambda value: "bfloat16" if value == normalized.dtype else str(value),
        raising=False,
    )
    result = classify_projection_outputs(
        {
            "normalized_hidden_owners": normalized,
            "projected_key_owners": projected,
            "current_key_owners": current,
        },
        np,
    )
    assert result["accepted_tpu_projection_match"] is True


def test_classifier_rejects_owner_disagreement() -> None:
    outputs = {
        "normalized_hidden_owners": np.zeros((2, 1, 6144), dtype=np.uint16),
        "projected_key_owners": np.zeros((2, 1, 128), dtype=np.float32),
        "current_key_owners": np.zeros((2, 1, 128), dtype=np.float32),
    }
    outputs["current_key_owners"][1, 0, 0] = 1.0
    result = classify_projection_outputs(outputs, np)
    assert result["accepted_tpu_projection_match"] is False
    assert result["outputs"]["current_key_owners"]["owners_equal"] is False


def test_classifier_rejects_extra_output() -> None:
    with pytest.raises(RuntimeError, match="catalogue"):
        classify_projection_outputs({"unexpected": np.zeros(1)}, np)


def test_wk_weight_rejects_wrong_identity() -> None:
    with pytest.raises(RuntimeError, match="weight drifted"):
        validate_wk_weight(np.zeros((2, 128, 6144), dtype=np.float32), np)
