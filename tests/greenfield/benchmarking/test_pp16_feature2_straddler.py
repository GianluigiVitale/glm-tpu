from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

import glm_tpu.greenfield.benchmarking.pp16_feature2_straddler as straddler
from glm_tpu.greenfield.benchmarking.pp16_feature2_straddler import (
    _RUNTIME_FEATURE_FILENAME,
    _RUNTIME_LAYER1_INPUT_NORM_NAME,
    _RUNTIME_LAYER1_INPUT_NORM_OFFSET,
    ACCEPTED_NORMALIZED_BITS,
    CURRENT_NORMALIZED_BITS,
    NORM_WEIGHT_BITS_AT_INDEX,
    _bf16_bits_to_float32,
    _finite_bf16_preimages,
    _float32_to_bf16_bits,
    _literal_double_round_outputs,
    _read_exact_bytes,
    _single_round_witness,
    _validate_hlo_float_type_correction,
    _validate_runtime_norm_weight_receipts,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)


def test_strategy_nd_model_axis_mapping_is_exact_nonidentity_inverse() -> None:
    model_axis_device_ids = tuple(
        int(item)
        for item in np.argsort(
            np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
        )
    )
    assert model_axis_device_ids != tuple(range(32))
    assert model_axis_device_ids == (
        0,
        8,
        16,
        24,
        2,
        10,
        18,
        26,
        4,
        12,
        20,
        28,
        6,
        14,
        22,
        30,
        1,
        9,
        17,
        25,
        3,
        11,
        19,
        27,
        5,
        13,
        21,
        29,
        7,
        15,
        23,
        31,
    )


def _runtime_norm_weight_runner(receipt_sha256: str) -> dict[str, object]:
    return {
        "selective_load": {
            "selected_reads": [
                {
                    "byte_count": 12_288,
                    "device_slot": device_slot,
                    "dtype": "BF16",
                    "filename": _RUNTIME_FEATURE_FILENAME.format(
                        device_slot=device_slot
                    ),
                    "name": _RUNTIME_LAYER1_INPUT_NORM_NAME,
                    "offset": _RUNTIME_LAYER1_INPUT_NORM_OFFSET,
                    "sha256": receipt_sha256,
                    "shape": [6144],
                }
                for device_slot in (0, 1)
            ]
        }
    }


def test_runtime_norm_weight_receipts_bind_both_slots_to_db550_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    norm_weight = np.arange(6144, dtype=np.uint16)
    norm_weight_sha256 = sha256(norm_weight.tobytes()).hexdigest()
    monkeypatch.setattr(
        straddler, "RUNTIME_LAYER1_INPUT_NORM_SHA256", norm_weight_sha256
    )
    report = _validate_runtime_norm_weight_receipts(
        _runtime_norm_weight_runner(norm_weight_sha256), norm_weight
    )
    assert report["device_slots"] == [0, 1]
    assert report["runtime_selected_read_count"] == 2
    assert report["runtime_selected_read_sha256"] == norm_weight_sha256


def test_runtime_norm_weight_receipts_reject_one_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    norm_weight = np.arange(6144, dtype=np.uint16)
    norm_weight_sha256 = sha256(norm_weight.tobytes()).hexdigest()
    monkeypatch.setattr(
        straddler, "RUNTIME_LAYER1_INPUT_NORM_SHA256", norm_weight_sha256
    )
    runner = _runtime_norm_weight_runner(norm_weight_sha256)
    assert isinstance(runner["selective_load"], dict)
    runner["selective_load"]["selected_reads"] = runner["selective_load"][
        "selected_reads"
    ][:1]
    with pytest.raises(BenchmarkValidationError, match="selected-read receipts"):
        _validate_runtime_norm_weight_receipts(runner, norm_weight)


def test_finite_bf16_preimages_are_exhaustive_for_negative_normal() -> None:
    preimages = _finite_bf16_preimages(47_953)
    assert preimages.dtype == np.float32
    assert preimages.size == 65_535
    assert np.all(_float32_to_bf16_bits(preimages) == np.uint16(47_953))
    assert float(preimages.min()) == pytest.approx(-0.0031967160757631063)
    assert float(preimages.max()) == pytest.approx(-0.0031814577523618937)


def test_literal_double_round_cannot_emit_protected_observed_bit() -> None:
    outputs = _literal_double_round_outputs(NORM_WEIGHT_BITS_AT_INDEX)
    assert CURRENT_NORMALIZED_BITS not in outputs
    assert ACCEPTED_NORMALIZED_BITS in outputs


def test_single_round_has_both_witnesses_for_same_bf16_row() -> None:
    carried = np.zeros((6144,), dtype=np.uint16)
    # Use the sealed local coordinate and enough nonzero centers to keep the
    # reference inverse in the same numerical regime as the protected row.
    carried[:] = np.uint16(15_255)
    carried[2795] = np.uint16(47_953)
    gamma = _bf16_bits_to_float32(
        np.asarray([NORM_WEIGHT_BITS_AT_INDEX], dtype=np.uint16)
    )[0]
    assert gamma == np.float32(0.0712890625)
    targets = set()
    preimages = _finite_bf16_preimages(47_953)
    center = _bf16_bits_to_float32(carried)
    other = np.float32(
        np.sum(center * center, dtype=np.float32)
        - np.float32(center[2795] * center[2795])
    )
    inverse = np.float32(1.0) / np.sqrt(
        (other + preimages * preimages) / np.float32(6144) + np.float32(1e-5),
        dtype=np.float32,
    )
    targets.update(
        int(value)
        for value in np.unique(_float32_to_bf16_bits(preimages * inverse * gamma))
    )
    assert {CURRENT_NORMALIZED_BITS, ACCEPTED_NORMALIZED_BITS} <= targets
    for target in (CURRENT_NORMALIZED_BITS, ACCEPTED_NORMALIZED_BITS):
        witness = _single_round_witness(
            carried,
            norm_weight_bits=NORM_WEIGHT_BITS_AT_INDEX,
            target_normalized_bits=target,
            hidden_index=2795,
        )
        assert witness["normalized_bits"] == target
        assert witness["rounded_carried_sha256"]


def test_hlo_marker_requires_scoped_bf16_correction() -> None:
    marker = (
        "%mul = f32[1,3072] multiply(%x, %w), metadata={"
        'op_name="greenfield_pp16_feature2_rms/mul"}, '
        'backend_config={"float_type_correction_info":'
        '{"original_type":"BF16"}}'
    )
    report = _validate_hlo_float_type_correction(marker)
    assert report["scoped_rms_marker_count"] == 1
    with pytest.raises(BenchmarkValidationError, match="lacks the scoped"):
        _validate_hlo_float_type_correction(marker.replace("BF16", "F32"))


def test_single_round_witness_rejects_wrong_geometry() -> None:
    with pytest.raises(ValueError, match=r"uint16\[6144\]"):
        _single_round_witness(
            np.zeros((3072,), dtype=np.uint16),
            norm_weight_bits=NORM_WEIGHT_BITS_AT_INDEX,
            target_normalized_bits=CURRENT_NORMALIZED_BITS,
            hidden_index=2795,
        )


def test_exact_buffer_survives_post_read_path_replacement(tmp_path) -> None:
    path = tmp_path / "sealed.bin"
    original = b"sealed-old-bytes"
    path.write_bytes(original)
    raw = _read_exact_bytes(path, sha256(original).hexdigest())
    path.write_bytes(b"hostile-new-bytes")
    assert raw == original
    with pytest.raises(BenchmarkValidationError, match="sealed input drifted"):
        _read_exact_bytes(path, sha256(original).hexdigest())
