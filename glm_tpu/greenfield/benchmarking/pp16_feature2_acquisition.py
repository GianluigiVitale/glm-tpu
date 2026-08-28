"""Authenticated source lineage for the PP16 feature2 HLO acquisition."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import BenchmarkValidationError
from ..validation.layer0_dsa_association import (
    inspect_greenfield_layer0_dsa_internal_observation,
)
from ..validation.short_context_dsa_oracle import (
    inspect_short_context_dsa_oracle,
)
from ..validation.short_context_oracle import inspect_short_context_oracle


FEATURE2_TOKEN_ORACLE_MANIFEST_SHA256 = (
    "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
)
FEATURE2_DSA_ORACLE_MANIFEST_SHA256 = (
    "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
)
FEATURE2_LAYER1_INTERNAL_REFERENCE_SHA256 = (
    "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054"
)
FEATURE2_DB529_INTERNAL_CONTRACT_SHA256 = (
    "9bdab5023b5775b787e15c3c76d542eab921bd7a704602b4502b2305fad03d4c"
)
FEATURE2_DB529_INTERNAL_TENSOR_SHA256 = (
    "c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b"
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def inspect_feature2_event1_lineage(
    *,
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    layer1_internal_reference: Path,
    db529_internal_dir: Path,
) -> dict[str, Any]:
    """Bind candidate event 1 to both accepted oracle and DB529 internals."""

    token_manifest = inspect_short_context_oracle(Path(token_oracle_dir))
    if (
        token_manifest.get("manifest_sha256")
        != FEATURE2_TOKEN_ORACLE_MANIFEST_SHA256
    ):
        raise BenchmarkValidationError("feature2 token oracle identity drifted")
    dsa_root = Path(dsa_oracle_dir)
    dsa_manifest = inspect_short_context_dsa_oracle(dsa_root)
    if (
        dsa_manifest.get("manifest_sha256")
        != FEATURE2_DSA_ORACLE_MANIFEST_SHA256
    ):
        raise BenchmarkValidationError("feature2 DSA oracle identity drifted")

    reference = Path(layer1_internal_reference)
    if (
        not reference.is_file()
        or _sha256_file(reference)
        != FEATURE2_LAYER1_INTERNAL_REFERENCE_SHA256
    ):
        raise BenchmarkValidationError(
            "feature2 layer-1 internal reference identity drifted"
        )
    expected_reference = {
        "accepted__current_key": ((128,), np.dtype(np.float32)),
        "accepted__head_weights": ((32,), np.dtype(np.float32)),
        "accepted__normalized_hidden": ((6144,), np.dtype(np.uint16)),
        "accepted__q_a_state": ((2048,), np.dtype(np.uint16)),
        "accepted__query": ((32, 128), np.dtype(np.float32)),
        "event_index": ((), np.dtype(np.int32)),
        "greenfield__current_key": ((128,), np.dtype(np.float32)),
        "greenfield__head_weights": ((32,), np.dtype(np.float32)),
        "greenfield__normalized_hidden": ((6144,), np.dtype(np.uint16)),
        "greenfield__q_a_state": ((2048,), np.dtype(np.uint16)),
        "greenfield__query": ((32, 128), np.dtype(np.float32)),
        "layer_id": ((), np.dtype(np.int32)),
        "position": ((), np.dtype(np.int32)),
    }
    with np.load(reference, allow_pickle=False) as handle:
        if set(handle.files) != set(expected_reference):
            raise BenchmarkValidationError(
                "feature2 layer-1 internal reference keys drifted"
            )
        layer1 = {
            name: np.asarray(handle[name]).copy() for name in handle.files
        }
    for name, (shape, dtype) in expected_reference.items():
        if layer1[name].shape != shape or layer1[name].dtype != dtype:
            raise BenchmarkValidationError(
                f"feature2 layer-1 internal field {name!r} drifted"
            )
    if (
        int(layer1["event_index"]) != 1
        or int(layer1["layer_id"]) != 1
        or int(layer1["position"]) != 8155
    ):
        raise BenchmarkValidationError(
            "feature2 layer-1 internal event identity drifted"
        )

    _, db529 = inspect_greenfield_layer0_dsa_internal_observation(
        Path(db529_internal_dir),
        expected_contract_sha256=FEATURE2_DB529_INTERNAL_CONTRACT_SHA256,
        expected_tensor_sha256=FEATURE2_DB529_INTERNAL_TENSOR_SHA256,
    )
    bindings = {
        "accepted__normalized_hidden": "normalized_hidden_bfloat16_bits",
        "accepted__q_a_state": "q_a_state_bfloat16_bits",
        "accepted__query": "query",
        "accepted__head_weights": "head_weights",
        "accepted__current_key": "current_key",
    }
    db529_mismatch_counts: dict[str, int] = {}
    for reference_name, db529_name in bindings.items():
        mismatch_count = int(
            np.count_nonzero(layer1[reference_name] != db529[db529_name][1])
        )
        db529_mismatch_counts[reference_name] = mismatch_count
        if mismatch_count == 0:
            raise BenchmarkValidationError(
                "feature2 DB529 mechanism source was substituted for the "
                f"accepted layer-1 target: {reference_name}"
            )
    if (
        int(db529["producer_layer_ids"][1]) != 1
        or int(db529["decode_position"][0]) != 8155
    ):
        raise BenchmarkValidationError("feature2 DB529 event mapping drifted")

    from safetensors import safe_open

    tensor_path = dsa_root / dsa_manifest["files"]["tensors"]["filename"]
    with safe_open(tensor_path, framework="np") as handle:
        producer_layers = handle.get_tensor("producer_layer_ids")
        positions = np.ascontiguousarray(
            handle.get_tensor("selected_positions")[0, 1]
        )
        scores = np.ascontiguousarray(
            handle.get_tensor("selected_scores")[0, 1]
        )
        valid_count = int(handle.get_tensor("valid_counts")[0, 1])
        decode_position = int(handle.get_tensor("decode_positions")[0])
    if (
        int(producer_layers[1]) != 1
        or positions.shape != (2048,)
        or scores.shape != (2048,)
        or valid_count != 2048
        or decode_position != 8155
    ):
        raise BenchmarkValidationError("feature2 DSA event-1 target drifted")
    return {
        "db529_internal_contract_sha256": (
            FEATURE2_DB529_INTERNAL_CONTRACT_SHA256
        ),
        "db529_mechanism_source_mismatch_counts": db529_mismatch_counts,
        "db529_internal_tensor_sha256": FEATURE2_DB529_INTERNAL_TENSOR_SHA256,
        "decode_position": decode_position,
        "dsa_oracle_manifest_sha256": FEATURE2_DSA_ORACLE_MANIFEST_SHA256,
        "event_index": 1,
        "expected_positions_sha256": _sha256_array(positions),
        "expected_scores_sha256": _sha256_array(scores),
        "expected_valid_count": valid_count,
        "layer1_current_key_sha256": _sha256_array(
            layer1["accepted__current_key"]
        ),
        "layer1_head_weights_sha256": _sha256_array(
            layer1["accepted__head_weights"]
        ),
        "layer1_internal_reference_sha256": (
            FEATURE2_LAYER1_INTERNAL_REFERENCE_SHA256
        ),
        "layer1_normalized_hidden_sha256": _sha256_array(
            layer1["accepted__normalized_hidden"]
        ),
        "layer1_q_a_state_sha256": _sha256_array(
            layer1["accepted__q_a_state"]
        ),
        "layer1_query_sha256": _sha256_array(layer1["accepted__query"]),
        "producer_layer_id": 1,
        "target_source": "sealed_layer1_reference_and_short_context_dsa_oracle",
        "token_oracle_manifest_sha256": FEATURE2_TOKEN_ORACLE_MANIFEST_SHA256,
    }
