from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    PP16_FEATURE2_CONTEXT_LENGTH,
    PP16_FEATURE2_MINIMUM_STATE_BYTES_PER_DEVICE,
    PP16_FEATURE2_N82_SCALE_SHA256,
    PP16_FEATURE2_N82_WEIGHT_SHA256,
    PP16_FEATURE2_RUNTIME_MANIFEST_SHA256,
    derive_feature2_tensor_allowlist,
    feature2_minimum_acquisition_roles,
    pp16_feature2_transport_pairs,
    read_feature2_owner_headers,
    split_feature2_bf16_bits,
    validate_feature2_acquisition_scope,
    validate_feature2_acquisition_reads,
    validate_feature2_accepted_state,
    validate_feature2_boundary_stablehlo,
    validate_feature2_n82_manifest,
    validate_feature2_transport_contract,
)
from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
)
from glm_tpu.greenfield.benchmarking.transport_chain import (
    TransportChainConfig,
    TransportKind,
)
from glm_tpu.greenfield.errors import (
    BenchmarkValidationError,
    HloContractViolationError,
)
from glm_tpu.greenfield.types import PlanName
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)


REAL_DB550_BOUNDARY = Path(
    "/home/gianl/gcs-models/results/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)


def _manifest() -> dict[str, object]:
    files = []
    for slot in range(2):
        filename = f"base_decoder_runtime_feature/stage_00/device_slot_0{slot}.safetensors"
        files.append(
            {
                "destination_filename": filename,
                "device_id": slot,
                "device_slot": slot,
                "stage_id": 0,
                "tensors": [
                    {
                        "byte_count": 32 * 6144 * 82,
                        "name": "attention.slot_01.qkv_a.weight_bits",
                        "sha256": PP16_FEATURE2_N82_WEIGHT_SHA256,
                        "sources": [
                            {
                                "selected_shape": [2048, 6144],
                                "source_device_slot": slot,
                                "source_filename": filename,
                                "source_shape": [2048, 6144],
                                "source_tensor_name": "attention.slot_01.q_a.weight_bits",
                                "source_tensor_sha256": (
                                    "3487ad2d9b2ff9d2bbf2c0405392d246"
                                    "6a5c3ccfe8017f617efbca0d9fbe25d7"
                                ),
                            },
                            {
                                "selected_shape": [576, 6144],
                                "source_device_slot": slot,
                                "source_filename": filename,
                                "source_shape": [576, 6144],
                                "source_tensor_name": "attention.slot_01.kv_a.weight_bits",
                                "source_tensor_sha256": (
                                    "6b68a46ff3615547d8ba5879df2d0e5e"
                                    "cb8562b830f93bff9e446d5488c1e4ff"
                                ),
                            },
                        ],
                        "transform": "fuse_qkv_a_output_shards",
                    },
                    {
                        "byte_count": 32 * 48 * 82 * 4,
                        "name": "attention.slot_01.qkv_a.scale_inv",
                        "sha256": PP16_FEATURE2_N82_SCALE_SHA256,
                        "sources": [
                            {
                                "selected_shape": [16, 48],
                                "source_device_slot": slot,
                                "source_filename": filename,
                                "source_shape": [16, 48],
                                "source_tensor_name": "attention.slot_01.q_a.scale_inv",
                                "source_tensor_sha256": (
                                    "4432012ede431b2741f739ac29cef64c"
                                    "fbee445544afe38586c1c03a5805721b"
                                ),
                            },
                            {
                                "selected_shape": [5, 48],
                                "source_device_slot": slot,
                                "source_filename": filename,
                                "source_shape": [5, 48],
                                "source_tensor_name": "attention.slot_01.kv_a.scale_inv",
                                "source_tensor_sha256": (
                                    "135aef9afce7e975c564cf48d2322ee1"
                                    "769f9dc0b089f27c19c406acac538059"
                                ),
                            },
                        ],
                        "transform": "fuse_qkv_a_expanded_scales",
                    },
                ],
            }
        )
    manifest = {
        "attention_projection_layout": "fused_qkv_a_virtual_tp32_n82_v1",
        "files": files,
        "model_id": "zai-org/GLM-5.2-FP8",
        "plan_id": "PP16_LP2",
    }
    manifest["manifest_sha256"] = sha256(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return manifest


def test_feature2_split_is_ordered_and_byte_exact() -> None:
    row = np.arange(6144, dtype=np.uint16).reshape(1, 6144)
    shards, record = split_feature2_bf16_bits(row)
    assert shards.shape == (2, 1, 3072)
    assert record.exact
    assert record.source_sha256 == record.concatenated_sha256
    assert np.array_equal(np.concatenate(tuple(shards), axis=1), row)
    with pytest.raises(BenchmarkValidationError, match="uint16"):
        split_feature2_bf16_bits(row.astype(np.int32))


def test_feature2_n82_owners_are_identical_and_owner_local() -> None:
    manifest = _manifest()
    report = validate_feature2_n82_manifest(
        manifest, expected_manifest_sha256=manifest["manifest_sha256"]
    )
    assert report["passed"]
    assert report["inferred_weight_shape"] == [32, 6144, 82]
    assert report["inferred_scale_shape"] == [32, 48, 82]
    bad = deepcopy(_manifest())
    bad["files"][1]["tensors"][0]["sources"][0]["source_device_slot"] = 0
    with pytest.raises(BenchmarkValidationError, match="owner-locally"):
        validate_feature2_n82_manifest(
            bad, expected_manifest_sha256=bad["manifest_sha256"]
        )
    stale_hash = deepcopy(_manifest())
    stale_hash["files"][0]["tensors"][0]["byte_count"] += 1
    with pytest.raises(BenchmarkValidationError, match="self-hash"):
        validate_feature2_n82_manifest(
            stale_hash,
            expected_manifest_sha256=stale_hash["manifest_sha256"],
        )


@pytest.mark.skipif(
    not REAL_DB550_BOUNDARY.is_file(), reason="protected DB550 boundary is unavailable"
)
def test_feature2_accepted_state_pins_both_rows_and_ordered_halves() -> None:
    with np.load(REAL_DB550_BOUNDARY, allow_pickle=False) as payload:
        model_axis_device_ids = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        dense, expected_carried = derive_expected_dense_boundary_bits(
            payload["dense_virtual_partials_bfloat16_bits"],
            payload["post_attention_residual_bfloat16_bits"],
            model_axis_device_ids,
        )
        input_residual = np.ascontiguousarray(
            payload["post_attention_residual_bfloat16_bits"]
        )
    packed, report = validate_feature2_accepted_state(dense, input_residual)
    assert packed.shape == (2, 2, 1, 3072)
    assert packed.dtype == np.uint16
    assert report["passed"]
    np.testing.assert_array_equal(
        np.concatenate(tuple(packed[:, 0]), axis=1), dense
    )
    np.testing.assert_array_equal(
        np.concatenate(tuple(packed[:, 1]), axis=1), input_residual
    )
    assert report["carried_output"]["source_sha256"] == sha256(
        expected_carried.tobytes(order="C")
    ).hexdigest()
    bad = dense.copy()
    bad[0, 0] ^= np.uint16(1)
    with pytest.raises(BenchmarkValidationError, match="source SHA"):
        validate_feature2_accepted_state(bad, input_residual)
    with pytest.raises(BenchmarkValidationError, match="input_residual source SHA"):
        validate_feature2_accepted_state(dense, expected_carried)


def test_feature2_real_manifest_derives_exact_range_allowlist() -> None:
    root = Path(
        "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
        "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_"
        "20260827T164842844148623Z"
    )
    manifest_path = root / "runtime_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("protected PP16 runtime manifest is unavailable")
    manifest = json.loads(manifest_path.read_text())
    headers = read_feature2_owner_headers(root, manifest)
    allowlist = derive_feature2_tensor_allowlist(manifest, headers)
    assert len(allowlist) == 78
    report = validate_feature2_acquisition_reads(
        allowlist, [item.to_dict() for item in allowlist]
    )
    assert report["bytes_by_slot"] == {0: 1_199_760_512, 1: 1_199_760_512}
    bad = [item.to_dict() for item in allowlist]
    bad.append(dict(bad[-1], name="dense.slot_01.down.weight_bits"))
    with pytest.raises(BenchmarkValidationError, match="exact manifest-derived"):
        validate_feature2_acquisition_reads(allowlist, bad)


def test_feature2_acquisition_scope_stops_before_layer1_output() -> None:
    report = validate_feature2_acquisition_scope(
        feature2_minimum_acquisition_roles(),
        context_length=PP16_FEATURE2_CONTEXT_LENGTH,
    )
    assert report["passed"]
    assert report["maximum_state_bytes_per_device"] == (
        PP16_FEATURE2_MINIMUM_STATE_BYTES_PER_DEVICE
    )
    with pytest.raises(BenchmarkValidationError, match="post-discriminator"):
        validate_feature2_acquisition_scope(
            (*feature2_minimum_acquisition_roles(), "layer1.attention_output"),
            context_length=PP16_FEATURE2_CONTEXT_LENGTH,
        )
    with pytest.raises(BenchmarkValidationError, match="exactly 8156"):
        validate_feature2_acquisition_scope(
            feature2_minimum_acquisition_roles(), context_length=2048
        )


def test_feature2_transport_reuses_two_slot_preserving_pp16_rings() -> None:
    config = TransportChainConfig(
        plan=PlanName.PP16_LP2,
        kind=TransportKind.DEVICE_RESIDENT,
        rows=2,
        width=3072,
        dtype="bfloat16",
        warmup_iterations=1,
        measured_iterations=1,
    )
    pairs = pp16_feature2_transport_pairs()
    report = validate_feature2_transport_contract(config, pairs)
    assert report == {
        "lane_payload_shape": [2, 1, 3072],
        "passed": True,
        "stage_payload_shape": [2, 1, 3072],
        "transport_pair_count": 32,
        "violations": [],
    }
    bad = list(pairs)
    bad[0], bad[1] = (bad[0][0], bad[1][1]), (bad[1][0], bad[0][1])
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_transport_contract(config, bad)


def test_feature2_transport_forced_32_has_two_live_bf16_lanes() -> None:
    program = r'''
import json
import re
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import lower_feature2_transport, validate_feature2_transport_stablehlo
from glm_tpu.greenfield.errors import HloContractViolationError

transport = lower_feature2_transport()
mutations = (
    transport.stablehlo.replace("[0, 2]", "[0, 3]", 1),
    transport.stablehlo.replace(
        "(tensor<2x1x3072xbf16>) -> tensor<2x1x3072xbf16>",
        "(tensor<2x1x3072xf32>) -> tensor<2x1x3072xf32>",
        1,
    ),
    transport.stablehlo + '\n"stablehlo.all_reduce"',
    transport.stablehlo.replace(
        '%11 = "stablehlo.collective_permute"(%10)',
        '%11 = "stablehlo.collective_permute"(%9)',
        1,
    ),
    transport.stablehlo.replace(
        '%13 = "stablehlo.collective_permute"(%12)',
        '%13 = "stablehlo.collective_permute"(%10)',
        1,
    ),
    re.sub(
        r"sdy\.return (%[0-9]+) : tensor<1x2x1x3072xbf16>",
        "sdy.return %arg1 : tensor<1x2x1x3072xbf16>",
        transport.stablehlo,
        count=1,
    ),
    transport.stablehlo.replace(
        "return %0 : tensor<32x2x1x3072xbf16>",
        "return %arg0 : tensor<32x2x1x3072xbf16>",
        1,
    ),
    transport.stablehlo.replace(
        'manual_axes={"device"}', 'manual_axes={"feature"}', 1
    ),
    transport.stablehlo.replace("3072", "6144", 1),
)
refused = []
for mutation in mutations:
    try:
        validate_feature2_transport_stablehlo(mutation)
    except HloContractViolationError:
        refused.append(True)
    else:
        refused.append(False)
print(json.dumps({"contract": dict(transport.stablehlo_contract), "refused": refused}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["refused"] == [True] * 9
    assert result["contract"] == {
        "collective_permute_count": 16,
        "lane_count": 2,
        "passed": True,
        "stage_count": 16,
        "stage_payload_shape": [2, 1, 3072],
        "violations": [],
    }


def test_feature2_boundary_forced_two_has_exact_stablehlo_contract() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import build_feature2_boundary

boundary = build_feature2_boundary()
carried, normalized = boundary.compiled(
    boundary.dense_update, boundary.carried_residual, boundary.norm_weight
)
result = {
    "carried_shape": list(carried.shape),
    "contract": dict(boundary.stablehlo_contract),
    "normalized_shape": list(normalized.shape),
}
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["carried_shape"] == [2, 1, 3072]
    assert result["normalized_shape"] == [1, 6144]
    assert result["contract"]["passed"]
    assert result["contract"]["all_reduce_count"] == 1
    assert result["contract"]["all_gather_count"] == 1


def test_feature2_embedding_forced_two_is_exact_and_never_builds_full_row() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    feature2_embedding_mapped,
)

mesh = Mesh(np.asarray(jax.devices()), ("feature",))
source = np.arange(2 * 4 * 6144, dtype=np.float32).reshape(2, 4, 6144)
source = np.asarray(np.sin(source / 29.0), dtype=jnp.bfloat16)
embedding = jax.device_put(
    source,
    NamedSharding(mesh, P("feature", None, None)),
)
def mapped(local_embedding, token):
    value = feature2_embedding_mapped(
        token,
        local_embedding[0],
        axis_name="feature",
        pairs=((0, 1), (1, 0)),
    )
    return value[None, ...]
execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(P("feature", None, None), P()),
    out_specs=P("feature", None, None),
    check_vma=False,
)
token = jnp.asarray(5, dtype=jnp.int32)
lowered = jax.jit(execute).lower(embedding, token)
actual = np.asarray(jax.jit(execute)(embedding, token))
concatenated = np.concatenate((actual[0], actual[1]), axis=-1)
stablehlo = lowered.as_text()
print(json.dumps({
    "collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
    "exact": bool(np.array_equal(
        concatenated.view(np.uint16), source[1, 1][None, :].view(np.uint16)
    )),
    "full_activation": "tensor<1x6144xbf16>" in stablehlo,
    "half_slice_count": stablehlo.count("tensor<1x3072xbf16>"),
    "shape": list(actual.shape),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["collective_permute_count"] == 1
    assert result["exact"]
    assert not result["full_activation"]
    assert result["half_slice_count"] > 0
    assert result["shape"] == [2, 1, 3072]


def test_feature2_stablehlo_mutations_refuse() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import build_feature2_boundary, validate_feature2_boundary_stablehlo
from glm_tpu.greenfield.errors import HloContractViolationError

boundary = build_feature2_boundary()
valid = boundary.stablehlo
mutations = (
    valid.replace("tensor<1x3072xbf16>) -> tensor<1x6144xbf16>", "tensor<1x3072xf32>) -> tensor<1x6144xf32>", 1),
    valid.replace("[[0, 1]]", "[[0, 2]]", 1),
    valid.replace("stablehlo.add %arg6, %arg7", "stablehlo.multiply %arg6, %arg7", 1),
    valid.replace(
        "applies stablehlo.add across dimensions = [0, 1]",
        "applies stablehlo.multiply across dimensions = [0, 1]",
        1,
    ),
    valid.replace(
        "%cst = stablehlo.constant dense<0.000000e+00>",
        "%cst = stablehlo.constant dense<1.000000e+00>",
        1,
    ),
    valid.replace(
        "applies stablehlo.add across dimensions = [0, 1]",
        "applies stablehlo.add across dimensions = [1]",
        1,
    ),
    valid.replace('"stablehlo.all_gather"(%17)', '"stablehlo.all_gather"(%16)', 1),
    valid.replace("sdy.return %19, %18", "sdy.return %19, %16", 1),
    valid.replace("stablehlo.multiply %6, %14", "stablehlo.add %6, %14", 1),
    valid.replace('manual_axes={"feature"}', 'manual_axes={"device"}', 1),
    valid + '\n"stablehlo.collective_permute"',
)
refused = []
for mutation in mutations:
    try:
        validate_feature2_boundary_stablehlo(mutation)
    except HloContractViolationError:
        refused.append(True)
    else:
        refused.append(False)
print(json.dumps({"refused": refused}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["refused"] == [True] * 11
