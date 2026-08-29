from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_program import (
    _update_liveness_digest,
    _validate_feature2_runtime_devices,
    load_feature2_prefill_runtime_inputs,
)
from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.kernels.reference.rotary import rotary_table_sha256

REAL_PP16_FEATURE2_RUNTIME = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP16_LP2/greenfield_runtime_feature_qkv_direct_pp16_"
    "20260827T164842844148623Z"
)
REAL_8K_ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)


@pytest.mark.skipif(
    not REAL_8K_ORACLE.is_dir(), reason="protected 8K oracle is unavailable"
)
def test_runtime_inputs_bind_tokens_pages_position_and_main_rope() -> None:
    inputs = load_feature2_prefill_runtime_inputs(REAL_8K_ORACLE)
    assert inputs.candidate_token_ids.shape == (8156,)
    assert int(inputs.candidate_token_ids[-1]) == 220
    np.testing.assert_array_equal(inputs.candidate_positions, np.arange(8156))
    np.testing.assert_array_equal(inputs.block_tables, np.arange(16)[None, :])
    np.testing.assert_array_equal(inputs.context_lengths, [8156])
    np.testing.assert_array_equal(inputs.current_position, [8155])
    assert inputs.main_rope_table.shape == (8192, 64)
    assert rotary_table_sha256(inputs.main_rope_table) == (
        "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
    )


def test_carried_liveness_digest_depends_on_values_owner_and_order() -> None:
    zero = jnp.zeros((2,), dtype=jnp.uint32)
    first = jnp.arange(3072, dtype=jnp.float32)[None, :].astype(jnp.bfloat16)
    second = (first + jnp.bfloat16(1)).astype(jnp.bfloat16)

    def digest(rows: tuple[jnp.ndarray, jnp.ndarray], owner: int) -> np.ndarray:
        value = zero
        for position, row in enumerate(rows):
            value = _update_liveness_digest(
                value,
                row,
                jnp.asarray(position, dtype=jnp.int32),
                jnp.asarray(owner, dtype=jnp.int32),
            )
        return np.asarray(value)

    ordered = digest((first, second), 0)
    assert not np.array_equal(ordered, digest((second, first), 0))
    assert not np.array_equal(ordered, digest((first, second), 1))
    changed = np.asarray(first).copy()
    changed[0, 17] = -3
    assert not np.array_equal(
        ordered,
        digest((jnp.asarray(changed), second), 0),
    )


def test_feature2_runtime_device_coordinates_normalize_without_weakening() -> None:
    adjacent = (
        SimpleNamespace(id=0, coords=[0, 0, 0]),
        SimpleNamespace(id=1, coords=[1, 0, 0]),
    )
    assert _validate_feature2_runtime_devices(adjacent) == adjacent
    with pytest.raises(PlanValidationError):
        _validate_feature2_runtime_devices(
            (
                SimpleNamespace(id=0, coords=[0, 0, 0]),
                SimpleNamespace(id=1, coords=[0, 1, 0]),
            )
        )
    with pytest.raises(PlanValidationError):
        _validate_feature2_runtime_devices(
            (
                SimpleNamespace(id=0, coords=[0, 0, 0]),
                SimpleNamespace(id=2, coords=[1, 0, 0]),
            )
        )


@pytest.mark.skipif(
    not REAL_PP16_FEATURE2_RUNTIME.is_dir() or not REAL_8K_ORACLE.is_dir(),
    reason="protected PP16 runtime or 8K oracle is unavailable",
)
def test_complete_program_abstract_graph_is_exact_lp2_and_fail_closed() -> None:
    program = r"""
from pathlib import Path
from hashlib import sha256
import json
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.benchmarking.pp16_feature2_prefill import (
    build_feature2_prefill_graph, load_feature2_prefill_inputs,
)
from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
    derive_feature2_tensor_allowlist, read_feature2_owner_headers,
)
from glm_tpu.greenfield.benchmarking.pp16_feature2_program import (
    _EXECUTABLE_WEIGHT_SHAPES, build_feature2_prefill_program,
    validate_feature2_prefill_jaxpr, validate_feature2_prefill_result_abstract,
    validate_feature2_executable_weight_contract,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError

root = Path(__import__('os').environ['FEATURE2_RUNTIME'])
oracle = Path(__import__('os').environ['FEATURE2_ORACLE'])
manifest = json.loads((root / 'runtime_manifest.json').read_text())
graph = build_feature2_prefill_graph(
    derive_feature2_tensor_allowlist(
        manifest, read_feature2_owner_headers(root, manifest)
    ),
    load_feature2_prefill_inputs(oracle),
)
built = build_feature2_prefill_program(graph, devices=jax.devices())
successor = build_feature2_prefill_program(
    graph,
    devices=jax.devices(),
    full_width_rounded_then_slice=True,
)
observer = build_feature2_prefill_program(
    graph,
    devices=jax.devices(),
    full_width_rounded_then_slice=True,
    observe_position_113=True,
)
db518 = build_feature2_prefill_program(
    graph,
    devices=jax.devices(),
    full_width_rounded_then_slice=True,
    observe_position_113=True,
    exact_layer0_prompt_keys=True,
)
dtypes = {'bf16': jnp.bfloat16, 'f32': jnp.float32, 'u8': jnp.uint8}
def abstract(shape, dtype, spec):
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=NamedSharding(built.mesh, spec)
    )
weights = {
    name: abstract((2, *shape), dtypes[dtype], built.weight_specs[name])
    for name, (shape, dtype) in _EXECUTABLE_WEIGHT_SHAPES.items()
}
query = abstract((2, 2048, 2048), jnp.float32, P('feature', None, None))
wk = abstract((2, 128, 6144), jnp.float32, P('feature', None, None))
arguments = (
    weights, query, query, wk, wk,
    abstract((8156,), jnp.int32, P()),
    abstract((8156,), jnp.int32, P()),
    abstract((1, 16), jnp.int32, P()),
    abstract((1,), jnp.int32, P()),
    abstract((1,), jnp.int32, P()),
    abstract((8192, 64), jnp.bfloat16, P()),
)
def inspect_program(value):
    text = str(jax.make_jaxpr(value.execute)(*arguments))
    contract = validate_feature2_prefill_jaxpr(
        text,
        full_width_rounded_then_slice=(
            value.full_width_rounded_then_slice
        ),
        observe_position_113=value.observe_position_113,
        exact_layer0_prompt_keys=value.exact_layer0_prompt_keys,
    )
    return {
        'contract': contract,
        'full_width_rounded_then_slice': value.full_width_rounded_then_slice,
        'observe_position_113': value.observe_position_113,
        'jaxpr_sha256': contract['jaxpr_sha256'],
        'raw_jaxpr_sha256': sha256(text.encode()).hexdigest(),
        'terminal': validate_feature2_prefill_result_abstract(
            jax.eval_shape(value.execute, *arguments),
            observe_position_113=value.observe_position_113,
        ),
    }, text

default_report, jaxpr = inspect_program(built)
successor_report, successor_jaxpr = inspect_program(successor)
observer_report, observer_jaxpr = inspect_program(observer)
db518_report, db518_jaxpr = inspect_program(db518)
mutations = (
    jaxpr + '\ndebug_callback',
    jaxpr.replace(
        'name=greenfield_pregathered_sparse_mla_h16_k2048_b512_w640',
        'name=greenfield_pregathered_sparse_mla_h32_k2048_b512_w640',
        1,
    ),
    jaxpr.replace('Precision.DEFAULT, Precision.HIGHEST', 'Precision.HIGHEST', 1),
    jaxpr + '\nbf16[32,6144]',
)
rejected = 0
for mutation in mutations:
    try:
        validate_feature2_prefill_jaxpr(mutation)
    except BenchmarkValidationError:
        rejected += 1
hybrid_rejected = False
try:
    validate_feature2_prefill_jaxpr(
        successor_jaxpr.replace(
            'name=greenfield_fp8_strategy_nd_o_m8_k512_n6144',
            'name=greenfield_fp8_strategy_nd_o_m8_k512_n3072',
        ),
        full_width_rounded_then_slice=True,
    )
except BenchmarkValidationError:
    hybrid_rejected = True
causal_mutations_rejected = 0
causal_mutations = (
    observer_jaxpr.replace(' 113', ' 114') + ('\n 113' * 4),
    observer_jaxpr.replace('pmin[', 'pmax[', 1) + '\npmin[',
)
for mutation in causal_mutations:
    try:
        validate_feature2_prefill_jaxpr(
            mutation,
            full_width_rounded_then_slice=True,
            observe_position_113=True,
        )
    except BenchmarkValidationError:
        causal_mutations_rejected += 1
cpu_mesh = "AbstractMesh('feature': 2, axis_types=(Manual,), device_kind=cpu, num_cores=None, platform=cpu)"
tpu_mesh = "AbstractMesh('feature': 2, axis_types=(Manual,), device_kind=TPU v4, num_cores=2, platform=tpu)"
observer_tpu_contract = validate_feature2_prefill_jaxpr(
    observer_jaxpr.replace(cpu_mesh, tpu_mesh),
    full_width_rounded_then_slice=True,
    observe_position_113=True,
)
db518_tpu_contract = validate_feature2_prefill_jaxpr(
    db518_jaxpr.replace(cpu_mesh, tpu_mesh),
    full_width_rounded_then_slice=True,
    observe_position_113=True,
    exact_layer0_prompt_keys=True,
)
runtime_mesh_mutations_rejected = 0
runtime_mesh_mutations = (
    observer_jaxpr.replace(cpu_mesh, tpu_mesh, 1),
    observer_jaxpr.replace('device_kind=cpu', 'device_kind=TPU v5'),
)
for mutation in runtime_mesh_mutations:
    try:
        validate_feature2_prefill_jaxpr(
            mutation,
            full_width_rounded_then_slice=True,
            observe_position_113=True,
        )
    except BenchmarkValidationError:
        runtime_mesh_mutations_rejected += 1
print(json.dumps({
    'causal_mutations_rejected': causal_mutations_rejected,
    'default': default_report,
    'db518': db518_report,
    'db518_tpu_contract': db518_tpu_contract,
    'graph_sha256': graph.graph_sha256,
    'hybrid_rejected': hybrid_rejected,
    'jaxpr_distinct': jaxpr != successor_jaxpr,
    'observer': observer_report,
    'observer_distinct': observer_jaxpr != successor_jaxpr,
    'observer_tpu_contract': observer_tpu_contract,
    'rejected_mutations': rejected,
    'runtime_mesh_mutations_rejected': runtime_mesh_mutations_rejected,
    'successor': successor_report,
    'weight_contract': validate_feature2_executable_weight_contract(graph),
    'weight_count': len(built.weight_specs),
}, sort_keys=True))
"""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["FEATURE2_RUNTIME"] = str(REAL_PP16_FEATURE2_RUNTIME)
    env["FEATURE2_ORACLE"] = str(REAL_8K_ORACLE)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["graph_sha256"] == (
        "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d"
    )
    assert result["weight_count"] == 37
    assert result["weight_contract"] == {
        "dense_final_leaf_count": 4,
        "dense_source_leaf_count": 6,
        "executable_leaf_count": 37,
        "passed": True,
        "source_leaf_count": 39,
    }
    assert result["rejected_mutations"] == 4
    assert result["hybrid_rejected"] is True
    assert result["causal_mutations_rejected"] == 2
    assert result["runtime_mesh_mutations_rejected"] == 2
    expected_contract = {
        "all_gather": 27,
        "convolution": 197,
        "fp8_attention_o_n3072": 128,
        "fp8_attention_o_n6144": 0,
        "forbidden_markers": [],
        "h16_b512_attention": 8,
        "jaxpr_canonicalizer_version": 1,
        "jaxpr_runtime_mesh": "cpu",
        "jaxpr_runtime_mesh_fragment_count": 2,
        "jaxpr_sha256": (
            "459866bc4d9bc0dc581aad1704e74db26e2d3576dec5b88979db5c09f96cd11b"
        ),
        "observation_position_literal": 0,
        "passed": True,
        "physical_m64_projection": 4,
        "pmin": 1,
        "ppermute": 12,
        "psum": 16,
        "raw_jaxpr_sha256": (
            "75deaf2087d62885eb6e0a9a4d26317ad70e405f912d793dbd9bc355de6d856d"
        ),
        "scan": 18,
    }
    expected_terminal = {
        "output_count": 15,
        "passed": True,
        "sealed_boundary_capture": True,
        "terminal_dtypes": [
            "int32",
            "int32",
            "float32",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "float32",
            "float32",
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "uint32",
            "bool",
        ],
        "terminal_shapes": [
            [1, 2048],
            [1],
            [1, 2048],
            [2, 1, 3072],
            [2, 1, 32, 256],
            [1, 576],
            [2, 1, 6144],
            [2, 1, 2048],
            [2, 1, 32, 128],
            [2, 1, 32],
            [2, 16, 256, 640],
            [2, 16, 256, 128],
            [2, 16, 256, 128],
            [2, 2],
            [1],
        ],
    }
    assert result["jaxpr_distinct"] is True
    assert result["observer_distinct"] is True
    assert result["default"] == {
        "contract": expected_contract,
        "full_width_rounded_then_slice": False,
        "observe_position_113": False,
        "jaxpr_sha256": "459866bc4d9bc0dc581aad1704e74db26e2d3576dec5b88979db5c09f96cd11b",
        "raw_jaxpr_sha256": "75deaf2087d62885eb6e0a9a4d26317ad70e405f912d793dbd9bc355de6d856d",
        "terminal": expected_terminal,
    }
    assert result["successor"] == {
        "contract": {
            **expected_contract,
            "convolution": 133,
            "fp8_attention_o_n3072": 0,
            "fp8_attention_o_n6144": 64,
            "jaxpr_sha256": (
                "7e1e4b549fa7b87c098f06644e918fe9f4fb05255a089ae9a94d1f218922aac0"
            ),
            "raw_jaxpr_sha256": (
                "9773c7b150a5b277116b33574b56f40316da24c5fc497d8827edbeb83fde372d"
            ),
        },
        "full_width_rounded_then_slice": True,
        "observe_position_113": False,
        "jaxpr_sha256": "7e1e4b549fa7b87c098f06644e918fe9f4fb05255a089ae9a94d1f218922aac0",
        "raw_jaxpr_sha256": "9773c7b150a5b277116b33574b56f40316da24c5fc497d8827edbeb83fde372d",
        "terminal": expected_terminal,
    }
    assert result["observer"] == {
        "contract": {
            **expected_contract,
            "convolution": 133,
            "fp8_attention_o_n3072": 0,
            "fp8_attention_o_n6144": 64,
            "jaxpr_sha256": (
                "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
            ),
            "observe_position_113": True,
            "observation_position_literal": 4,
            "position113_causal_contract": {
                "counter_carried_through_all_scans": True,
                "observed_value_count": 8,
                "position": 113,
                "terminal_validity_gated_by_exact_count": True,
            },
            "pmin": 2,
            "raw_jaxpr_sha256": (
                "a6ce2233eed467ae85be0a718532f3e4996b1588673b45687173459caa5adbf0"
            ),
        },
        "full_width_rounded_then_slice": True,
        "jaxpr_sha256": (
            "c8b59417193eac580290648c28430cd8c11477dc6f465585c347b2001e97d1cd"
        ),
        "raw_jaxpr_sha256": (
            "a6ce2233eed467ae85be0a718532f3e4996b1588673b45687173459caa5adbf0"
        ),
        "observe_position_113": True,
        "terminal": {
            **expected_terminal,
            "observe_position_113": True,
            "output_count": 24,
            "terminal_dtypes": [
                *expected_terminal["terminal_dtypes"],
                "bfloat16",
                "bfloat16",
                "float32",
                "float32",
                "float32",
                "int32",
                "int32",
                "float32",
                "int32",
            ],
            "terminal_shapes": [
                *expected_terminal["terminal_shapes"],
                [2, 1, 6144],
                [2, 1, 2048],
                [2, 1, 32, 128],
                [2, 1, 32],
                [2, 1, 128],
                [2, 1, 2048],
                [2, 1],
                [2, 1, 2048],
                [2, 1],
            ],
        },
    }
    db518_contract = {
        **result["observer"]["contract"],
        "all_gather": 31,
        "exact_layer0_prompt_keys": True,
        "jaxpr_sha256": (
            "f87c0f162523bf6221d2824aa4b7268f1c770a55d2f8cde6da7506f595e9447f"
        ),
        "physical_m64_projection": 8,
        "ppermute": 16,
        "psum": 20,
        "raw_jaxpr_sha256": (
            "f2c8b06727e2839109d78626a9eb94639747beab3800b34a4392c4a865b47aef"
        ),
        "scan": 22,
    }
    assert result["db518"] == {
        "contract": db518_contract,
        "full_width_rounded_then_slice": True,
        "jaxpr_sha256": (
            "f87c0f162523bf6221d2824aa4b7268f1c770a55d2f8cde6da7506f595e9447f"
        ),
        "observe_position_113": True,
        "raw_jaxpr_sha256": (
            "f2c8b06727e2839109d78626a9eb94639747beab3800b34a4392c4a865b47aef"
        ),
        "terminal": result["observer"]["terminal"],
    }
    observer_tpu_contract = result["observer_tpu_contract"]
    assert observer_tpu_contract["jaxpr_runtime_mesh"] == "tpu_v4"
    assert observer_tpu_contract["raw_jaxpr_sha256"] == (
        "4e7f821d11e9a4fcf12ae39ef657c2a3f054c0a5976898d62069470e4d957d5d"
    )
    assert {
        key: value
        for key, value in observer_tpu_contract.items()
        if key not in {"jaxpr_runtime_mesh", "raw_jaxpr_sha256"}
    } == {
        key: value
        for key, value in result["observer"]["contract"].items()
        if key not in {"jaxpr_runtime_mesh", "raw_jaxpr_sha256"}
    }
    db518_tpu_contract = result["db518_tpu_contract"]
    assert db518_tpu_contract["jaxpr_runtime_mesh"] == "tpu_v4"
    assert db518_tpu_contract["raw_jaxpr_sha256"] == (
        "2a81016ac0db10e47beca9a86bb858193b43131f5e61aa5f339a98d7bfdb866b"
    )
    assert {
        key: value
        for key, value in db518_tpu_contract.items()
        if key not in {"jaxpr_runtime_mesh", "raw_jaxpr_sha256"}
    } == {
        key: value
        for key, value in db518_contract.items()
        if key not in {"jaxpr_runtime_mesh", "raw_jaxpr_sha256"}
    }
