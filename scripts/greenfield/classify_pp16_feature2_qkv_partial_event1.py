#!/usr/bin/env python3
"""Reject or admit the LP2 split-K arm at coherent layer-1/event-1 on CPU."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

import ml_dtypes
import numpy as np
from safetensors.numpy import load_file


RUNTIME_MANIFEST_FILE_SHA256 = (
    "e13ccefb7341756cd68d85e51209eaa8516ea506eac7ace3b4fd0a1d56828032"
)
RUNTIME_MANIFEST_SELF_SHA256 = (
    "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab"
)
DB518_FILE_SHA256 = "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0"
DB518_COMPARISON_FILE_SHA256 = (
    "06ee82b9d487e3fdf8f9f19d4e824e33f1f9d453738a0090ace5e2cac7272a4d"
)
ACCEPTED_INTERNAL_FILE_SHA256 = (
    "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054"
)
ADMISSION_FILE_SHA256 = (
    "33e8dd0a1b0fb8e9c46c1acb3ce05357beff8969fb694464f32f6a071493619c"
)
DSA_MANIFEST_FILE_SHA256 = (
    "62c3fc2ad45d368c91cc41901947a69abf7a8c7abb57fb56b088871e90e5f1d3"
)
DSA_MANIFEST_SELF_SHA256 = (
    "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
)
DSA_TENSOR_FILE_SHA256 = (
    "b591a4622a8c646799989f235bc98bcf8ae99e9e04d2ad55a982d70ba00dde82"
)
INTENDED_Q_A_SHA256 = "c488a3f95ecb476adcc2ea8d0d8db1b6a88944fcfaccb9a85042a3a34b06368c"
DB518_LAYER1_POST_EVENT_CACHE_SHA256 = (
    "adc7cc27d46c654f4df11959bd3e13ef52502a8167976ea94838b426a30ea728"
)
DB518_LAYER1_PROMPT_HISTORY_SHA256 = (
    "f1594a300ad7593254eb96b2874a3bb6f1b265da681c08ff00ea96dda7068477"
)
SEALED_LAYER0_ACCEPTED_PROMPT_CACHE_SHA256 = (
    "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
)
EXPECTED_ARMS_SHA256 = (
    "6068183397d415f2baa28e33c53d0229a970d6173bad61e9763fba2c1d93382b"
)
EXPECTED_TOOLCHAIN = {
    "jax": "0.10.1",
    "jaxlib": "0.10.1",
    "ml_dtypes": "0.5.4",
    "numpy": "2.3.5",
    "safetensors": "0.7.0",
}
EXPECTED_SOURCE_SHA256 = {
    "glm_tpu/greenfield/benchmarking/pp16_feature2_qkv_partial_event1.py": (
        "714104ece85756e93687566eaa5e657eb3a779c1f2c65d6fb56fa10c1b8ca82c"
    ),
    "glm_tpu/greenfield/kernels/stage_local.py": (
        "0fa80ae74390a79bb63e6ac82357a08770b644bc07093139a825eb68b82ac490"
    ),
    "glm_tpu/greenfield/kernels/reference/attention.py": (
        "0314931e50dedb9b40b2570c7716b38b32421a8c8749eb36bb5636717787c54a"
    ),
    "glm_tpu/greenfield/kernels/reference/dsa.py": (
        "c4b451ab7ca2bf79b7cc7b148a7b1b146996051891c1c82f1952d8ab9ef53fac"
    ),
    "glm_tpu/greenfield/kernels/reference/dsa_association.py": (
        "28d2faf69c9619ee38c4d5dfdbd8cd4a5618f90bb2993aabee5fc9fe777355f5"
    ),
    "glm_tpu/greenfield/kernels/reference/fp8.py": (
        "9d4ded1d4d553d103ff64eb5066507a612aee183985fe45ee62cd1fe3c4b9f58"
    ),
    "glm_tpu/greenfield/kernels/reference/linear.py": (
        "c3d679dab63d1975f0fbfd1ceda7bd8e198621face135151f5fe05cedd07a0f6"
    ),
    "glm_tpu/greenfield/kernels/reference/prefill_index.py": (
        "74dbda42b55bd7190f283e33b182d22d12b8ae73eadf75877ab26e71493fa086"
    ),
    "glm_tpu/greenfield/kernels/reference/qkv_a.py": (
        "1e6d2cb6464d1733708c227283898f0687c965e1883cbea4c441ece2a40d3bb3"
    ),
    "glm_tpu/greenfield/kernels/reference/rmsnorm.py": (
        "fc0234da4ea4e718cd42d3a3ebb529de3b77d453c13590fa931d834654efe572"
    ),
    "glm_tpu/greenfield/kernels/reference/rotary.py": (
        "cb17440803331de962328b409212e9647fc736590c3b525a4a2d43daa46b05b3"
    ),
}
EXPECTED_CPU_REJECTION = {
    "accepted_positions_mismatches": 1892,
    "accepted_set_difference": 18,
    "control_positions_sha256": (
        "788f4ffcb228fe24faf4c417d4c82f743cd4f4968a0c6c951b3afeb06e571af6"
    ),
    "control_vs_db518_positions_mismatches": 1723,
    "intended_positions_sha256": (
        "3e724627906caf35ac0054578d34dffcb345a8f44e9e2225acbf8aa92fcad600"
    ),
    "prompt_only_cache_sha256": (
        "bbee9a907725d9375141b38d6908fdd206d935bb9b1bf2b4cbea7ddaa93bb023"
    ),
}
_TENSOR_NAMES = (
    "attention.slot_01.input_norm",
    "attention.slot_01.qkv_a.weight_bits",
    "attention.slot_01.qkv_a.scale_inv",
    "attention.slot_01.q_a_norm",
    "indexer.slot_01.wq_b.weight_bits",
    "indexer.slot_01.wq_b.scale_inv",
    "indexer.slot_01.wk.weight_bits",
    "indexer.slot_01.wk.scale_inv",
    "indexer.slot_01.key_norm_weight",
    "indexer.slot_01.key_norm_bias",
    "indexer.slot_01.head_weight",
)
_DTYPES = {
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F32": np.dtype("<f4"),
    "U8": np.dtype("u1"),
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return sha256(payload).hexdigest()


def _verify_toolchain(jax: Any) -> dict[str, str]:
    import jaxlib
    import safetensors

    observed = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "ml_dtypes": ml_dtypes.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
    }
    if observed != EXPECTED_TOOLCHAIN:
        raise RuntimeError(f"event-1 classifier toolchain drifted: {observed!r}")
    return observed


def _verify_source_files() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    observed = {name: _file_sha256(root / name) for name in EXPECTED_SOURCE_SHA256}
    if observed != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("event-1 replay source identity drifted")
    return observed


def _verify_db518_comparison(path: Path, cache_bits: np.ndarray) -> dict[str, Any]:
    if _file_sha256(path) != DB518_COMPARISON_FILE_SHA256:
        raise RuntimeError("DB518 sealed comparison SHA-256 drifted")
    comparison = json.loads(path.read_text())
    full_width = comparison.get("full_width_comparison", {})
    captured = full_width.get("captured_array_sha256", {})
    position113 = comparison.get("position113_comparison", {})
    if (
        comparison.get("capture_sha256") != DB518_FILE_SHA256
        or captured.get("layer1_index_cache_owners_bfloat16_bits")
        != DB518_LAYER1_POST_EVENT_CACHE_SHA256
        or _array_sha256(cache_bits) != DB518_LAYER1_POST_EVENT_CACHE_SHA256
        or position113.get("capture_npz_sha256") != DB518_FILE_SHA256
        or position113.get("prompt_cache_matches_accepted") is not True
        or position113.get("prompt_cache_diff", {}).get("mismatch_count") != 0
        or position113.get("accepted_prompt_cache_sha256")
        != SEALED_LAYER0_ACCEPTED_PROMPT_CACHE_SHA256
        or position113.get("candidate_prompt_cache_sha256")
        != SEALED_LAYER0_ACCEPTED_PROMPT_CACHE_SHA256
    ):
        raise RuntimeError("DB518 sealed cache provenance drifted")
    return {
        "comparison_file_sha256": DB518_COMPARISON_FILE_SHA256,
        "layer0_prompt_cache_exact": True,
        "layer0_prompt_cache_sha256": SEALED_LAYER0_ACCEPTED_PROMPT_CACHE_SHA256,
        "layer1_post_event_cache_sha256": DB518_LAYER1_POST_EVENT_CACHE_SHA256,
        "layer1_prompt_history_accepted_exactness": "UNKNOWN_NO_ACCEPTED_LAYER1_CACHE",
    }


def _require_regular(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} is unavailable") from error
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"{label} must be one regular file")
    return value


def _read_runtime(
    root: Path,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    manifest_path = root / "runtime_manifest.json"
    success_path = root / "SUCCESS"
    if _file_sha256(manifest_path) != RUNTIME_MANIFEST_FILE_SHA256:
        raise RuntimeError("runtime manifest file SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != RUNTIME_MANIFEST_SELF_SHA256:
        raise RuntimeError("runtime manifest self identity drifted")
    if success_path.read_text() != (
        f"{RUNTIME_MANIFEST_SELF_SHA256}  runtime_manifest.json\n"
    ):
        raise RuntimeError("runtime SUCCESS marker drifted")
    records = {
        int(record["device_slot"]): record
        for record in manifest.get("files", ())
        if record.get("stage_id") == 0 and record.get("device_slot") in (0, 1)
    }
    if set(records) != {0, 1}:
        raise RuntimeError("runtime lacks its exact stage-0 LP2 owners")

    owners: list[dict[str, np.ndarray]] = []
    receipts: list[dict[str, Any]] = []
    for slot in (0, 1):
        record = records[slot]
        path = root / str(record["destination_filename"])
        observed_stat = _require_regular(path, f"runtime owner {slot}")
        if observed_stat.st_size != int(record["file_bytes"]):
            raise RuntimeError(f"runtime owner {slot} file size drifted")
        header_bytes = int(record["header_bytes"])
        with path.open("rb") as handle:
            raw_header = handle.read(header_bytes)
            if sha256(raw_header).hexdigest() != record["header_sha256"]:
                raise RuntimeError(f"runtime owner {slot} header drifted")
            encoded_length = struct.unpack("<Q", raw_header[:8])[0]
            if encoded_length + 8 != header_bytes:
                raise RuntimeError(f"runtime owner {slot} header length drifted")
            header = json.loads(raw_header[8:])
            tensor_records = {item["name"]: item for item in record["tensors"]}
            values: dict[str, np.ndarray] = {}
            for name in _TENSOR_NAMES:
                layout = header.get(name)
                tensor = tensor_records.get(name)
                if not isinstance(layout, dict) or not isinstance(tensor, dict):
                    raise RuntimeError(f"runtime owner {slot} lacks {name}")
                dtype = _DTYPES.get(layout.get("dtype"))
                if dtype is None:
                    raise RuntimeError(f"runtime owner {slot} {name} dtype drifted")
                start, end = (int(item) for item in layout["data_offsets"])
                shape = tuple(int(item) for item in layout["shape"])
                byte_count = end - start
                if byte_count != int(np.prod(shape)) * dtype.itemsize or (
                    byte_count != tensor["byte_count"]
                ):
                    raise RuntimeError(f"runtime owner {slot} {name} bytes drifted")
                handle.seek(header_bytes + start)
                payload = handle.read(byte_count)
                observed_sha = sha256(payload).hexdigest()
                if len(payload) != byte_count or observed_sha != tensor["sha256"]:
                    raise RuntimeError(f"runtime owner {slot} {name} payload drifted")
                values[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
                receipts.append(
                    {
                        "byte_count": byte_count,
                        "device_slot": slot,
                        "name": name,
                        "sha256": observed_sha,
                        "shape": list(shape),
                    }
                )
        owners.append(values)
    for name in (
        "attention.slot_01.input_norm",
        "attention.slot_01.qkv_a.weight_bits",
        "attention.slot_01.qkv_a.scale_inv",
        "attention.slot_01.q_a_norm",
        "indexer.slot_01.wk.weight_bits",
        "indexer.slot_01.wk.scale_inv",
        "indexer.slot_01.key_norm_weight",
        "indexer.slot_01.key_norm_bias",
    ):
        if not np.array_equal(owners[0][name], owners[1][name]):
            raise RuntimeError(f"replicated runtime tensor {name} differs by owner")
    return owners, receipts


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected = np.ascontiguousarray(expected)
    observed = np.ascontiguousarray(observed)
    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        raise RuntimeError("comparison shape or dtype drifted")
    mismatches = np.flatnonzero(expected.reshape(-1) != observed.reshape(-1))
    return {
        "first_mismatch_indices": mismatches[:16].tolist(),
        "mismatch_count": int(mismatches.size),
        "observed_sha256": _array_sha256(observed),
    }


def _event_comparison(
    expected_positions: np.ndarray,
    expected_counts: np.ndarray,
    expected_scores: np.ndarray,
    positions: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    expected_set = set(np.asarray(expected_positions).reshape(-1).tolist())
    observed_set = set(np.asarray(positions).reshape(-1).tolist())
    return {
        "positions": _comparison(expected_positions, positions),
        "scores": _comparison(expected_scores, scores),
        "symmetric_set_difference_count": len(expected_set ^ observed_set),
        "valid_counts": _comparison(expected_counts, counts),
    }


def classify(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("event-1 classifier requires JAX_PLATFORMS=cpu")

    import jax
    import jax.numpy as jnp

    from glm_tpu.greenfield.benchmarking.pp16_feature2_qkv_partial_event1 import (
        PP16_EVENT1_LOCAL_ROWS_PER_PAGE,
        PP16_EVENT1_LOGICAL_PAGE_SIZE,
        PP16_EVENT1_POSITION,
        build_pp16_event1_cpu_replay,
        derive_db518_prompt_only_cache_bits,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        decode_stage_local_prefill_index_wk_bf16,
        promote_stage_local_prefill_index_wk,
    )
    from glm_tpu.greenfield.kernels.reference.qkv_a import (
        FusedQkvAContract,
        one_row_fused_qkv_a_convolution,
    )

    if jax.default_backend() != "cpu" or len(jax.devices()) != 2:
        raise RuntimeError("event-1 classifier requires exactly two CPU devices")
    toolchain = _verify_toolchain(jax)
    source_sha256 = _verify_source_files()
    if _file_sha256(args.admission) != ADMISSION_FILE_SHA256:
        raise RuntimeError("split-K CPU admission artifact drifted")
    admission = json.loads(args.admission.read_text())
    if admission.get("classification") != (
        "FP32_PARTIAL_ARM_CPU_NECESSARY_CONDITION_PASSES;"
        "TPU_PHYSICAL_ASSOCIATION_AND_EVENT1_EXACTNESS_UNRESOLVED"
    ):
        raise RuntimeError("split-K CPU admission classification drifted")

    owners, receipts = _read_runtime(args.runtime_root)
    if _file_sha256(args.db518_result) != DB518_FILE_SHA256:
        raise RuntimeError("DB518 result SHA-256 drifted")
    with np.load(args.db518_result, allow_pickle=False) as values:
        normalized_owners_bits = np.ascontiguousarray(
            values["current_normalized_hidden_owners_bfloat16_bits"]
        )
        captured_q_owners_bits = np.ascontiguousarray(
            values["current_q_a_state_owners_bfloat16_bits"]
        )
        post_event_cache_bits = np.ascontiguousarray(
            values["layer1_index_cache_owners_bfloat16_bits"]
        )
        db518_positions = np.ascontiguousarray(values["event1_positions"])
        db518_counts = np.ascontiguousarray(values["event1_valid_counts"])
        db518_scores = np.ascontiguousarray(values["event1_scores"])
        db518_query = np.ascontiguousarray(values["current_dsa_query_owners"])[0]
        db518_head = np.ascontiguousarray(values["current_dsa_head_weights_owners"])[0]
    cache_provenance = _verify_db518_comparison(
        args.db518_comparison, post_event_cache_bits
    )
    if not np.array_equal(normalized_owners_bits[0], normalized_owners_bits[1]) or (
        not np.array_equal(captured_q_owners_bits[0], captured_q_owners_bits[1])
    ):
        raise RuntimeError("DB518 current owner observations differ")

    if _file_sha256(args.accepted_internals) != ACCEPTED_INTERNAL_FILE_SHA256:
        raise RuntimeError("accepted layer-1 internals drifted")
    with np.load(args.accepted_internals, allow_pickle=False) as values:
        accepted_normalized_bits = np.ascontiguousarray(
            values["accepted__normalized_hidden"]
        )
        accepted_q_bits = np.ascontiguousarray(values["accepted__q_a_state"])
        accepted_query = np.ascontiguousarray(values["accepted__query"])[None, ...]
        accepted_head = np.ascontiguousarray(values["accepted__head_weights"])[
            None, ...
        ]
        accepted_key = np.ascontiguousarray(values["accepted__current_key"])[None, ...]
        if (
            int(values["event_index"]) != 1
            or int(values["layer_id"]) != 1
            or (int(values["position"]) != 8155)
        ):
            raise RuntimeError("accepted layer-1 internal event identity drifted")

    dsa_manifest_path = args.dsa_oracle / "manifest.json"
    dsa_tensor_path = args.dsa_oracle / "dsa_events.safetensors"
    if _file_sha256(dsa_manifest_path) != DSA_MANIFEST_FILE_SHA256 or (
        _file_sha256(dsa_tensor_path) != DSA_TENSOR_FILE_SHA256
    ):
        raise RuntimeError("accepted DSA oracle files drifted")
    dsa_manifest = json.loads(dsa_manifest_path.read_text())
    if dsa_manifest.get("manifest_sha256") != DSA_MANIFEST_SELF_SHA256:
        raise RuntimeError("accepted DSA oracle identity drifted")
    dsa = load_file(dsa_tensor_path)
    for name, record in dsa_manifest["arrays"].items():
        value = np.ascontiguousarray(dsa[name])
        if (
            list(value.shape) != record["shape"]
            or str(value.dtype) != record["dtype"]
            or (_array_sha256(value) != record["sha256"])
        ):
            raise RuntimeError(f"accepted DSA oracle array {name} drifted")
    if dsa["decode_positions"][0] != 8155 or dsa["producer_layer_ids"][1] != 1:
        raise RuntimeError("accepted event-1 alignment drifted")
    accepted_positions = np.ascontiguousarray(dsa["selected_positions"][0, 1:2])
    accepted_scores = np.ascontiguousarray(dsa["selected_scores"][0, 1:2])
    accepted_counts = np.ascontiguousarray(dsa["valid_counts"][0, 1:2])

    expected_key_bits = accepted_key.astype(ml_dtypes.bfloat16).view(np.uint16)
    prompt_cache_bits, prompt_cache_sha = derive_db518_prompt_only_cache_bits(
        post_event_cache_bits, expected_key_bits
    )
    history_positions = np.arange(PP16_EVENT1_POSITION, dtype=np.int64)
    history_page_rows = history_positions % PP16_EVENT1_LOGICAL_PAGE_SIZE
    prompt_history_bits = np.ascontiguousarray(
        prompt_cache_bits[
            history_page_rows // PP16_EVENT1_LOCAL_ROWS_PER_PAGE,
            history_positions // PP16_EVENT1_LOGICAL_PAGE_SIZE,
            history_page_rows % PP16_EVENT1_LOCAL_ROWS_PER_PAGE,
        ]
    )
    prompt_history_sha = _array_sha256(prompt_history_bits)
    if prompt_history_sha != DB518_LAYER1_PROMPT_HISTORY_SHA256:
        raise RuntimeError("DB518 layer-1 prompt history drifted")

    materialized_query = []
    materialized_wk = []
    for owner in owners:
        materialized_query.append(
            np.asarray(
                dequantize_fp8_bits_block_weight(
                    jnp.asarray(owner["indexer.slot_01.wq_b.weight_bits"]),
                    jnp.asarray(owner["indexer.slot_01.wq_b.scale_inv"]),
                    output_dtype=jnp.float32,
                )
            )
        )
        wk_bf16 = decode_stage_local_prefill_index_wk_bf16(
            jnp.asarray(owner["indexer.slot_01.wk.weight_bits"]),
            jnp.asarray(owner["indexer.slot_01.wk.scale_inv"]),
        )
        materialized_wk.append(
            np.asarray(promote_stage_local_prefill_index_wk(wk_bf16))
        )

    normalized = normalized_owners_bits[0].view(ml_dtypes.bfloat16)
    captured_q = captured_q_owners_bits[0].view(ml_dtypes.bfloat16)
    intended_q = np.asarray(
        one_row_fused_qkv_a_convolution(
            jnp.asarray(normalized),
            jnp.asarray(owners[0]["attention.slot_01.qkv_a.weight_bits"]),
            jnp.asarray(owners[0]["attention.slot_01.qkv_a.scale_inv"]),
            jnp.asarray(owners[0]["attention.slot_01.q_a_norm"]),
            contract=FusedQkvAContract(),
        ).q_residual
    )
    if _array_sha256(intended_q) != INTENDED_Q_A_SHA256:
        raise RuntimeError("intended split-K CPU q-a state drifted")

    replay = build_pp16_event1_cpu_replay(devices=jax.devices())

    def owner_stack(name: str) -> np.ndarray:
        return np.stack([owner[name] for owner in owners])

    common = (
        jnp.asarray(prompt_cache_bits.view(ml_dtypes.bfloat16)),
        jnp.asarray(owner_stack("indexer.slot_01.wq_b.weight_bits")),
        jnp.asarray(owner_stack("indexer.slot_01.wq_b.scale_inv")),
        jnp.asarray(np.stack(materialized_query)),
        jnp.asarray(owner_stack("indexer.slot_01.wk.weight_bits")),
        jnp.asarray(owner_stack("indexer.slot_01.wk.scale_inv")),
        jnp.asarray(np.stack(materialized_wk)),
        jnp.asarray(owner_stack("indexer.slot_01.head_weight")),
        jnp.asarray(owners[0]["attention.slot_01.input_norm"]),
        jnp.asarray(owners[0]["attention.slot_01.q_a_norm"]),
        jnp.asarray(owner_stack("indexer.slot_01.key_norm_weight")),
        jnp.asarray(owner_stack("indexer.slot_01.key_norm_bias")),
    )

    def run_arm(
        name: str, one_normalized: np.ndarray, q_a: np.ndarray
    ) -> dict[str, Any]:
        result = replay(jnp.asarray(one_normalized), jnp.asarray(q_a), *common)
        host = {field: np.asarray(getattr(result, field)) for field in result._fields}
        if not bool(np.all(host["contract_valid"])):
            raise RuntimeError(f"event-1 replay arm {name} failed its device contract")
        cache_bits = np.ascontiguousarray(host["index_cache"]).view(np.uint16)
        current_page = PP16_EVENT1_POSITION // PP16_EVENT1_LOGICAL_PAGE_SIZE
        current_page_row = PP16_EVENT1_POSITION % PP16_EVENT1_LOGICAL_PAGE_SIZE
        current_owner = current_page_row // PP16_EVENT1_LOCAL_ROWS_PER_PAGE
        current_local_row = current_page_row % PP16_EVENT1_LOCAL_ROWS_PER_PAGE
        cache_current = cache_bits[
            current_owner, current_page, current_local_row
        ].copy()
        replay_current_key_bits = (
            np.ascontiguousarray(host["current_key"].astype(ml_dtypes.bfloat16))
            .view(np.uint16)
            .reshape(-1)
        )
        if not np.array_equal(cache_current, replay_current_key_bits):
            raise RuntimeError(
                f"event-1 replay arm {name} cache write differs from its current key"
            )
        cache_without_current = cache_bits.copy()
        cache_without_current[current_owner, current_page, current_local_row] = (
            np.uint16(0)
        )
        if not np.array_equal(cache_without_current, prompt_cache_bits):
            raise RuntimeError(
                f"event-1 replay arm {name} changed historical cache rows"
            )
        return {
            "accepted_event": _event_comparison(
                accepted_positions,
                accepted_counts,
                accepted_scores,
                host["selected_positions"],
                host["valid_counts"],
                host["selected_scores"],
            ),
            "accepted_internals": {
                "current_key": _comparison(accepted_key, host["current_key"]),
                "head_weights": _comparison(accepted_head, host["head_weights"]),
                "query": _comparison(accepted_query, host["query"]),
            },
            "db518_event": _event_comparison(
                db518_positions,
                db518_counts,
                db518_scores,
                host["selected_positions"],
                host["valid_counts"],
                host["selected_scores"],
            ),
            "db518_internals": {
                "head_weights": _comparison(db518_head, host["head_weights"]),
                "query": _comparison(db518_query, host["query"]),
            },
            "current_cache_key_vs_accepted": _comparison(
                expected_key_bits.reshape(-1), cache_current
            ),
            "current_cache_key_vs_replay_round": _comparison(
                replay_current_key_bits, cache_current
            ),
            "normalized_hidden_sha256": _array_sha256(one_normalized),
            "q_a_state_sha256": _array_sha256(q_a),
        }

    arms = {
        "accepted_current_row_sensitivity": run_arm(
            "accepted_current_row_sensitivity",
            accepted_normalized_bits.view(ml_dtypes.bfloat16)[None, :],
            accepted_q_bits.view(ml_dtypes.bfloat16)[None, :],
        ),
        "db518_captured_q_control": run_arm(
            "db518_captured_q_control", normalized, captured_q
        ),
        "intended_split_k_association": run_arm(
            "intended_split_k_association", normalized, intended_q
        ),
    }
    arms_sha256 = _canonical_json_sha256(arms)
    if arms_sha256 != EXPECTED_ARMS_SHA256:
        raise RuntimeError("complete frozen CPU event-1 observation drifted")
    control = arms["db518_captured_q_control"]["db518_event"]
    intended = arms["intended_split_k_association"]["accepted_event"]
    control_exact = all(
        control[name]["mismatch_count"] == 0
        for name in ("positions", "scores", "valid_counts")
    )
    intended_exact = all(
        intended[name]["mismatch_count"] == 0
        for name in ("positions", "scores", "valid_counts")
    )
    classification = (
        "CPU_EVENT1_ADMISSION_PASSES"
        if control_exact and intended_exact
        else "CPU_EVENT1_ADMISSION_REJECTED;NO_TPU_SUCCESSOR"
    )
    frozen_observation = {
        "accepted_positions_mismatches": intended["positions"]["mismatch_count"],
        "accepted_set_difference": intended["symmetric_set_difference_count"],
        "control_positions_sha256": control["positions"]["observed_sha256"],
        "control_vs_db518_positions_mismatches": control["positions"]["mismatch_count"],
        "intended_positions_sha256": intended["positions"]["observed_sha256"],
        "prompt_only_cache_sha256": prompt_cache_sha,
    }
    if frozen_observation != EXPECTED_CPU_REJECTION or control_exact or intended_exact:
        raise RuntimeError("frozen CPU event-1 rejection drifted")
    return {
        "accepted_internal_file_sha256": ACCEPTED_INTERNAL_FILE_SHA256,
        "admission_artifact_sha256": ADMISSION_FILE_SHA256,
        "arms": arms,
        "arms_sha256": arms_sha256,
        "artifact_kind": "pp16_feature2_qkv_khalf_event1_cpu_admission_v2",
        "classification": classification,
        "cpu_backend": jax.default_backend(),
        "cpu_control_exact": control_exact,
        "cpu_device_count": len(jax.devices()),
        "db518_comparison_file_sha256": DB518_COMPARISON_FILE_SHA256,
        "db518_result_file_sha256": DB518_FILE_SHA256,
        "dsa_manifest_file_sha256": DSA_MANIFEST_FILE_SHA256,
        "dsa_manifest_self_sha256": DSA_MANIFEST_SELF_SHA256,
        "dsa_tensor_file_sha256": DSA_TENSOR_FILE_SHA256,
        "gate_d_closed": False,
        "historical_cache_provenance": {
            **cache_provenance,
            "layer1_prompt_history_sha256": prompt_history_sha,
            "prompt_only_physical_cache_sha256": prompt_cache_sha,
        },
        "performance_claim": False,
        "prompt_only_cache_sha256": prompt_cache_sha,
        "protected_tpu_evidence": False,
        "runtime_manifest_file_sha256": RUNTIME_MANIFEST_FILE_SHA256,
        "runtime_manifest_self_sha256": RUNTIME_MANIFEST_SELF_SHA256,
        "runtime_tensor_receipts": receipts,
        "script_sha256": _file_sha256(Path(__file__)),
        "source_sha256": source_sha256,
        "status": "SUCCESS",
        "toolchain": toolchain,
        "tpu_execution_authorized": False,
        "tpus_used": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--db518-result", type=Path, required=True)
    parser.add_argument("--db518-comparison", type=Path, required=True)
    parser.add_argument("--accepted-internals", type=Path, required=True)
    parser.add_argument("--dsa-oracle", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("event-1 classifier requires JAX_PLATFORMS=cpu")
    if args.output.is_symlink() or args.output.exists():
        raise FileExistsError(f"classifier output already exists: {args.output}")
    report = classify(args)
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            args.output.unlink()
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
