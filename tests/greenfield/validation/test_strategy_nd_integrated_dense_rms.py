from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import array_sha256
from glm_tpu.greenfield.benchmarking.integrated_dense_rms import (
    NATIVE_SOURCE_ATTENDED_LATENT_KEY,
    NATIVE_SOURCE_ATTENDED_LATENT_SHA256,
    NATIVE_SOURCE_NPZ_SHA256,
    NATIVE_SOURCE_TOKEN_IDS,
    POST_ATTENTION_NORM_RAW_SHA256,
    _raw_sha256,
    load_integrated_dense_rms_inputs,
    model_axis_weights_to_physical,
    native_source_inputs,
    validate_integrated_checkpoint_success,
)
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import (
    INTEGRATED_DENSE_ACCEPTED_SOURCE_STABLEHLO_SHA256,
    INTEGRATED_DENSE_ORDINAL_RMS_STABLEHLO_SHA256,
    INTEGRATED_DENSE_PREDENSE_SPLIT_RMS_STABLEHLO_SHA256,
    INTEGRATED_DENSE_RMS_STABLEHLO_SHA256,
    INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256,
    _validate_split_predense_value_flow,
    integrated_dense_rms_hlo_policy,
    validate_integrated_dense_rms_hlo,
    validate_integrated_dense_rms_stablehlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import (
    HloContractPolicy,
    lint_hlo,
    parse_hlo_module,
)
from glm_tpu.greenfield.validation.strategy_nd_integrated_dense_rms import (
    CHECKPOINT_SUCCESS_SHA256,
    _recompute_comparison,
    _validate_capture,
    _validate_hlo_prevalidation,
    _validate_native_source_files,
    validate_strategy_nd_integrated_dense_rms,
)


REPO = Path(__file__).resolve().parents[3]
REAL_SOURCE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_NATIVE_SOURCE = Path(
    "/home/gianl/gcs-models/results/"
    "greenfield_layer0_attention_arithmetic_20260812T114701365714147Z"
)
REAL_STAGE0_SLOT0 = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP8_LP4/greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z/"
    "base_decoder_runtime_feature/stage_00/device_slot_00.safetensors"
)
REAL_SPLIT_TPU_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_INTEGRATED_SPLIT_TPU_HLO",
        "/home/gianl/glm-run/"
        "greenfield_recover_integrated_split_hlo_20260814T192500000000000Z/"
        "recovery_hlo/worker0/"
        "module_0012.jit_integrated.cl_914450892.after_codegen.txt",
    )
)
REAL_SPLIT_TPU_HLO_SHA256 = (
    "212aa36a9587ff390e6b0c18b654187d96e158eb896eaece1af51cda35df4e27"
)
REAL_ORDINAL_TPU_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_INTEGRATED_ORDINAL_TPU_HLO",
        "/home/gianl/glm-run/"
        "greenfield_strategy_nd_integrated_dense_ordinal_rms_"
        "20260814T193832987684744Z/hlo/"
        "strategy_nd_integrated_dense_ordinal_rms_bfloat16_32x6144."
        "optimized_hlo.txt",
    )
)
REAL_ORDINAL_TPU_HLO_SHA256 = (
    "433d4938b398769f4db442a7ce5af3baa365fc2137850233f0e9df8162dd042b"
)
REAL_ACCEPTED_SOURCE_TPU_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_INTEGRATED_ACCEPTED_SOURCE_TPU_HLO",
        "/home/gianl/glm-run/"
        "greenfield_strategy_nd_integrated_dense_accepted_source_context_"
        "20260814T212452244910289Z/recovered_hlo/"
        "strategy_nd_integrated_dense_accepted_source_context_"
        "bfloat16_32x6144.optimized_hlo.txt",
    )
)
REAL_ACCEPTED_SOURCE_TPU_HLO_SHA256 = (
    "081d1b1f3609085a2b455357f8f6f9186c7bb3c5c218c1d10c734bea2b0163f8"
)
REAL_LEGACY_INTEGRATED_RUN = Path(
    "/home/gianl/glm-run/"
    "greenfield_strategy_nd_integrated_dense_rms_20260814T174146122417710Z"
)
REAL_CHECKPOINT_ROOT = REAL_STAGE0_SLOT0.parents[2]


def _synthetic_predense_gate_hlo() -> str:
    return r'''HloModule predense_gate, num_partitions=32

%f32_add (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(%x, %y)
}

%predense_reduce (attention: bf16[1,6144], residual: bf16[1,6144]) -> f32[32] {
  %attention = bf16[1,6144]{1,0} parameter(0)
  %residual = bf16[1,6144]{1,0} parameter(1)
  %zero_bf16 = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{1,0} pad(%attention, %zero_bf16), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{1,0} pad(%residual, %zero_bf16), padding=0_31x0_0
  %attention_f32 = f32[32,6144]{1,0} convert(%attention_pad)
  %residual_f32 = f32[32,6144]{1,0} convert(%residual_pad)
  %sum = f32[32,6144]{1,0} add(%attention_f32, %residual_f32)
  %square = f32[32,6144]{1,0} multiply(%sum, %sum)
  %zero = f32[] constant(0)
  ROOT %reduce = f32[32]{0} reduce(%square, %zero), dimensions={1}, to_apply=%f32_add
}

%predense_rsqrt (reduced: f32[32]) -> f32[32] {
  %reduced = f32[32]{0} parameter(0)
  %scale = f32[] constant(0.000162760422)
  %scale_wide = f32[32]{0} broadcast(%scale), dimensions={}
  %mean = f32[32]{0} multiply(%reduced, %scale_wide)
  %epsilon = f32[] constant(1e-05)
  %epsilon_wide = f32[32]{0} broadcast(%epsilon), dimensions={}
  %variance = f32[32]{0} add(%mean, %epsilon_wide)
  ROOT %inverse = f32[32]{0} rsqrt(%variance)
}

%predense_norm (attention: bf16[1,6144], residual: bf16[1,6144], inverse: f32[32], weight: bf16[6144]) -> bf16[32,6144] {
  %attention = bf16[1,6144]{1,0} parameter(0)
  %residual = bf16[1,6144]{1,0} parameter(1)
  %inverse = f32[32]{0} parameter(2)
  %weight = bf16[6144]{0} parameter(3)
  %zero_bf16 = bf16[] constant(0)
  %attention_pad = bf16[32,6144]{1,0} pad(%attention, %zero_bf16), padding=0_31x0_0
  %residual_pad = bf16[32,6144]{1,0} pad(%residual, %zero_bf16), padding=0_31x0_0
  %attention_f32 = f32[32,6144]{1,0} convert(%attention_pad)
  %residual_f32 = f32[32,6144]{1,0} convert(%residual_pad)
  %sum = f32[32,6144]{1,0} add(%attention_f32, %residual_f32)
  %inverse_wide = f32[32,6144]{1,0} broadcast(%inverse), dimensions={0}
  %normalized = f32[32,6144]{1,0} multiply(%sum, %inverse_wide)
  %normalized_bf16 = bf16[32,6144]{1,0} convert(%normalized)
  %normalized_f32 = f32[32,6144]{1,0} convert(%normalized_bf16)
  %weight_wide = bf16[32,6144]{1,0} broadcast(%weight), dimensions={1}
  %weight_f32 = f32[32,6144]{1,0} convert(%weight_wide)
  %weighted = f32[32,6144]{1,0} multiply(%normalized_f32, %weight_f32)
  ROOT %weighted_bf16 = bf16[32,6144]{1,0} convert(%weighted)
}

%gate_body (attention: bf16[1,6144], residual: bf16[1,6144], inverse: f32[32], norm: bf16[6144], gate_weight: bf16[6144,768]) -> bf16[32,1,768] {
  %attention = bf16[1,6144]{1,0} parameter(0)
  %residual = bf16[1,6144]{1,0} parameter(1)
  %inverse = f32[32]{0} parameter(2)
  %norm = bf16[6144]{0} parameter(3)
  %gate_weight = bf16[6144,768]{1,0} parameter(4)
  %gate_lhs = bf16[32,6144]{1,0} fusion(%attention, %residual, %inverse, %norm), kind=kLoop, calls=%predense_norm
  %gate = f32[32,768]{1,0} convolution(%gate_lhs, %gate_weight), dim_labels=bf_io->bf
  %gate_bf16 = bf16[32,768]{1,0} convert(%gate)
  ROOT %gate_result = bf16[32,1,768]{2,1,0} bitcast(%gate_bf16)
}

ENTRY %main (p0: bf16[1,6144], p1: bf16[1,6144], p2: bf16[6144], p3: f8e4m3fn[1,1,6144,768], p4: f32[1,1,48,768], p5: f8e4m3fn[1,1,384,6144], p6: f32[1,1,3,6144], p7: bf16[6144]) -> bf16[32,1,768] {
  %p0 = bf16[1,6144]{1,0} parameter(0)
  %p1 = bf16[1,6144]{1,0} parameter(1)
  %p2 = bf16[6144]{0} parameter(2)
  %p3 = f8e4m3fn[1,1,6144,768]{3,2,1,0} parameter(3)
  %p4 = f32[1,1,48,768]{3,2,1,0} parameter(4)
  %p5 = f8e4m3fn[1,1,384,6144]{3,2,1,0} parameter(5)
  %p6 = f32[1,1,3,6144]{3,2,1,0} parameter(6)
  %p7 = bf16[6144]{0} parameter(7)
  %scheduled = f32[32]{0} fusion(%p0, %p1), kind=kLoop, calls=%predense_reduce
  %inverse = f32[32]{0} fusion(%scheduled), kind=kLoop, calls=%predense_rsqrt
  %zero = bf16[] constant(0)
  %gate_weight = bf16[6144,768]{1,0} broadcast(%zero), dimensions={}
  ROOT %gate_call = bf16[32,1,768]{2,1,0} fusion(%p0, %p1, %inverse, %p2, %gate_weight), kind=kOutput, calls=%gate_body
}
'''


def _synthetic_predense_report(hlo: str) -> object:
    return lint_hlo(
        parse_hlo_module(hlo),
        HloContractPolicy(
            name="synthetic-predense-gate",
            total_devices=32,
            repeated_region_patterns=(),
            maximum_repeated_collective_group_size=32,
            forbidden_row_width_pairs=(),
            require_repeated_region=False,
        ),
    )


def test_model_axis_weights_are_bijectively_mapped_to_physical_ids() -> None:
    value = np.arange(32 * 2, dtype=np.int32).reshape(32, 2)
    mapping = tuple(reversed(range(32)))
    physical = model_axis_weights_to_physical(value, mapping)
    for source_rank, physical_id in enumerate(mapping):
        assert np.array_equal(physical[physical_id], value[source_rank])
    with pytest.raises(ValueError, match="bijectively cover"):
        model_axis_weights_to_physical(value, tuple([0] * 32))


def test_native_source_graph_abstractly_traces_all_thirteen_inputs() -> None:
    code = r'''
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.benchmarking.integrated_dense_rms import _native_source_context_function

mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("member",))
mapped = jax.shard_map(
    _native_source_context_function(),
    mesh=mesh,
    in_specs=(
        P(), P(), P(),
        P("member", None, None), P("member", None, None),
        P("member", None, None), P("member", None, None),
        P("member", None, None), P("member", None, None, None),
        P("member", None, None, None), P("member", None, None, None),
        P("member", None, None, None), P(),
    ),
    out_specs=P(),
    check_vma=False,
)
arguments = (
    jax.ShapeDtypeStruct((64, 512), jnp.bfloat16),
    jax.ShapeDtypeStruct((32,), jnp.int32),
    jax.ShapeDtypeStruct((6144,), jnp.bfloat16),
    jax.ShapeDtypeStruct((32, 4840, 6144), jnp.bfloat16),
    jax.ShapeDtypeStruct((32, 7168, 512), jnp.uint8),
    jax.ShapeDtypeStruct((32, 56, 4), jnp.float32),
    jax.ShapeDtypeStruct((32, 512, 6144), jnp.uint8),
    jax.ShapeDtypeStruct((32, 4, 48), jnp.float32),
    jax.ShapeDtypeStruct((32, 1, 6144, 768), jnp.float8_e4m3fn),
    jax.ShapeDtypeStruct((32, 1, 48, 768), jnp.float32),
    jax.ShapeDtypeStruct((32, 1, 384, 6144), jnp.float8_e4m3fn),
    jax.ShapeDtypeStruct((32, 1, 3, 6144), jnp.float32),
    jax.ShapeDtypeStruct((6144,), jnp.bfloat16),
)
output = jax.eval_shape(mapped, *arguments)
assert output.shape == (1, 6144)
assert output.dtype == jnp.uint16
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=32",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    source = (
        REPO
        / "glm_tpu/greenfield/benchmarking/integrated_dense_rms.py"
    ).read_text()
    start = source.index("def _native_source_context_function()")
    end = source.index("\ndef build_integrated_dense_rms(", start)
    native = source[start:end]
    projection = native.index(
        'with jax.named_scope("native_source_context_attention_projection")'
    )
    finite_guard = native.index(
        "embedding_is_finite = jnp.isfinite(embedding_m32[0, 0])"
    )
    guarded_select = native.index(
        "local_attention = lax.select(", finite_guard
    )
    collective = native.index(
        'with jax.named_scope("native_source_context_attention_collective")'
    )
    assert projection < finite_guard < guarded_select < collective


@pytest.mark.skipif(
    not REAL_NATIVE_SOURCE.is_dir()
    or not REAL_SOURCE.is_file()
    or not REAL_STAGE0_SLOT0.is_file(),
    reason="protected native/integrated source absent",
)
def test_real_native_source_is_exactly_pinned_and_mutations_refuse(
    tmp_path: Path,
) -> None:
    from safetensors import safe_open

    for name in (
        "attention_arithmetic.npz",
        "remote_objects.json",
        "runner.json",
        "SUCCESS",
        "summary.json",
    ):
        shutil.copy2(REAL_NATIVE_SOURCE / name, tmp_path / name)
    native_path = _validate_native_source_files(tmp_path)
    assert sha256(native_path.read_bytes()).hexdigest() == NATIVE_SOURCE_NPZ_SHA256

    with safe_open(REAL_STAGE0_SLOT0, framework="np") as handle:
        post_norm = np.ascontiguousarray(
            handle.get_tensor("attention.slot_00.post_norm")
        )
    base = load_integrated_dense_rms_inputs(REAL_SOURCE, post_norm)
    with np.load(native_path, allow_pickle=False) as payload:
        latent = np.ascontiguousarray(payload[NATIVE_SOURCE_ATTENDED_LATENT_KEY])
    assert _raw_sha256(latent) == NATIVE_SOURCE_ATTENDED_LATENT_SHA256
    native = native_source_inputs(latent, base)
    assert np.array_equal(native.token_ids, NATIVE_SOURCE_TOKEN_IDS)

    changed = latent.copy()
    changed.view(np.uint8)[0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="attended latent drifted"):
        native_source_inputs(changed, base)
    (tmp_path / "rogue").write_text("not part of the sealed source")
    with pytest.raises(ValueError, match="native source file set drifted"):
        _validate_native_source_files(tmp_path)


@pytest.mark.skipif(
    not REAL_SOURCE.is_file() or not REAL_STAGE0_SLOT0.is_file(),
    reason="protected DB548/checkpoint source absent",
)
def test_real_integrated_source_is_exactly_pinned() -> None:
    from safetensors import safe_open

    with safe_open(REAL_STAGE0_SLOT0, framework="np") as handle:
        post_norm = np.ascontiguousarray(
            handle.get_tensor("attention.slot_00.post_norm")
        )
    assert __import__("hashlib").sha256(post_norm.tobytes()).hexdigest() == (
        POST_ATTENTION_NORM_RAW_SHA256
    )
    inputs = load_integrated_dense_rms_inputs(
        REAL_SOURCE,
        post_norm,
    )
    assert inputs.accepted_layer1_bits.shape == (6144,)


def test_exact_forced_32_cpu_stablehlo_and_mutation_refusal() -> None:
    code = r'''
from glm_tpu.greenfield.benchmarking.integrated_dense_rms import build_integrated_dense_rms
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import validate_integrated_dense_rms_stablehlo

compiled = build_integrated_dense_rms(tuple(range(32)), validate_hlo=False)
contract = validate_integrated_dense_rms_stablehlo(compiled.stablehlo)
assert contract["passed"]
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import integrated_dense_rms_hlo_policy, _validate_accepted_attention_input, _validate_preceding_attention_input
from glm_tpu.greenfield.benchmarking.dense_rms_replay import _validate_exact_dense_rms_value_flow
from glm_tpu.greenfield.sharding.hlo_contract import lint_hlo, parse_hlo_module
report = lint_hlo(parse_hlo_module(compiled.optimized_hlo), integrated_dense_rms_hlo_policy(tuple(range(32))))
assert report.valid, [item.to_dict() for item in report.violations]
try:
    validate_integrated_dense_rms_stablehlo(compiled.stablehlo.replace("9.99999974E-6", "2.99999974E-6", 1))
except ValueError:
    pass
else:
    raise AssertionError("StableHLO mutation was accepted")
print(contract["exact_graph_sha256"])
split = build_integrated_dense_rms(
    tuple(range(32)), validate_hlo=False, split_layer1_rms=True
)
split_contract = validate_integrated_dense_rms_stablehlo(
    split.stablehlo, split_layer1_rms=True
)
assert split_contract["passed"] and split_contract["split_layer1_rms"] is True
assert split.split_layer1_rms is True
try:
    validate_integrated_dense_rms_stablehlo(split.stablehlo)
except ValueError:
    pass
else:
    raise AssertionError("split StableHLO passed the tuple-schedule contract")
print(split_contract["exact_graph_sha256"])
ordinal = build_integrated_dense_rms(
    tuple(range(32)),
    validate_hlo=False,
    split_layer1_rms=True,
    preceding_attention_collective=True,
)
ordinal_contract = validate_integrated_dense_rms_stablehlo(
    ordinal.stablehlo,
    split_layer1_rms=True,
    preceding_attention_collective=True,
)
assert ordinal_contract["passed"]
assert ordinal.preceding_attention_collective is True
ordinal_report = lint_hlo(
    parse_hlo_module(ordinal.optimized_hlo),
    integrated_dense_rms_hlo_policy(
        tuple(range(32)), preceding_attention_collective=True
    ),
)
assert ordinal_report.valid, [item.to_dict() for item in ordinal_report.violations]
attention = tuple(
    item for item in ordinal_report.module.collectives
    if "integrated_dense_rms_preceding_attention_collective"
    in (item.op_name or "").split("/")
)
assert len(attention) == 1
assert _validate_preceding_attention_input(
    ordinal_report, attention[0]
)["exact_attention_collective_input"]
for old, new in (
    ("%constant.1.clone.2 = s32[] constant(0)", "%constant.1.clone.2 = s32[] constant(1)"),
    ("select(%select_n.11, %convert.166, %broadcast.50)", "select(%select_n.11, %broadcast.50, %broadcast.50)"),
    ("select(%select_n.11, %convert.166, %broadcast.50)", "select(%select_n.11, %broadcast.50, %convert.166)"),
    ("%broadcast_select_fusion = f32[32,6144]{1,0} fusion(", "%broadcast_select_fusion = f32[32,6144]{0,1} fusion("),
    ("%broadcast.50 = f32[32,6144]{1,0} broadcast(", "%broadcast.50 = f32[32,6144]{0,1} broadcast("),
    ('direction=EQ, metadata={op_name="jit(integrated)/shard_map/integrated_dense_rms_preceding_attention_collective/eq"', 'direction=NE, metadata={op_name="jit(integrated)/shard_map/integrated_dense_rms_preceding_attention_collective/eq"'),
):
    mutation = ordinal.optimized_hlo.replace(old, new, 1)
    assert mutation != ordinal.optimized_hlo
    mutated_report = lint_hlo(
        parse_hlo_module(mutation),
        integrated_dense_rms_hlo_policy(
            tuple(range(32)), preceding_attention_collective=True
        ),
    )
    mutated_attention = tuple(
        item for item in mutated_report.module.collectives
        if "integrated_dense_rms_preceding_attention_collective"
        in (item.op_name or "").split("/")
    )
    try:
        _validate_preceding_attention_input(
            mutated_report, mutated_attention[0]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("preceding attention mutation was accepted")
equivalent = ordinal.optimized_hlo.replace(
    'direction=EQ, metadata={op_name="jit(integrated)/shard_map/integrated_dense_rms_preceding_attention_collective/eq"',
    'direction=NE, metadata={op_name="jit(integrated)/shard_map/integrated_dense_rms_preceding_attention_collective/eq"',
    1,
).replace(
    "select(%select_n.11, %convert.166, %broadcast.50)",
    "select(%select_n.11, %broadcast.50, %convert.166)",
    1,
)
assert equivalent != ordinal.optimized_hlo
from jaxlib import xla_client
xla_client._xla.hlo_module_from_text(equivalent)
equivalent_report = lint_hlo(
    parse_hlo_module(equivalent),
    integrated_dense_rms_hlo_policy(
        tuple(range(32)), preceding_attention_collective=True
    ),
)
equivalent_attention = tuple(
    item for item in equivalent_report.module.collectives
    if "integrated_dense_rms_preceding_attention_collective"
    in (item.op_name or "").split("/")
)
assert _validate_preceding_attention_input(
    equivalent_report, equivalent_attention[0]
)["exact_attention_collective_input"]
equivalent_zero_layout = equivalent.replace(
    "%broadcast.50 = f32[32,6144]{1,0} broadcast(",
    "%broadcast.50 = f32[32,6144]{0,1} broadcast(",
    1,
)
assert equivalent_zero_layout != equivalent
equivalent_zero_report = lint_hlo(
    parse_hlo_module(equivalent_zero_layout),
    integrated_dense_rms_hlo_policy(
        tuple(range(32)), preceding_attention_collective=True
    ),
)
equivalent_zero_attention = tuple(
    item for item in equivalent_zero_report.module.collectives
    if "integrated_dense_rms_preceding_attention_collective"
    in (item.op_name or "").split("/")
)
try:
    _validate_preceding_attention_input(
        equivalent_zero_report, equivalent_zero_attention[0]
    )
except ValueError:
    pass
else:
    raise AssertionError("NE zero-branch layout mutation was accepted")
print(ordinal_contract["exact_graph_sha256"])
predense = build_integrated_dense_rms(
    tuple(range(32)),
    validate_hlo=False,
    split_layer1_rms=True,
    split_predense_rms=True,
)
predense_contract = validate_integrated_dense_rms_stablehlo(
    predense.stablehlo,
    split_layer1_rms=True,
    split_predense_rms=True,
)
assert predense_contract["passed"]
assert predense.split_predense_rms is True
assert predense_contract["split_predense_rms"] is True
try:
    validate_integrated_dense_rms_stablehlo(
        predense.stablehlo, split_layer1_rms=True
    )
except ValueError:
    pass
else:
    raise AssertionError("pre-dense split StableHLO passed the old split arm")
for forbidden in (
    {"split_layer1_rms": False, "split_predense_rms": True},
    {
        "split_layer1_rms": True,
        "split_predense_rms": True,
        "preceding_attention_collective": True,
    },
):
    try:
        build_integrated_dense_rms(
            tuple(range(32)), validate_hlo=False, **forbidden
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pre-dense split mode combination passed")
print(predense_contract["exact_graph_sha256"])
accepted = build_integrated_dense_rms(
    tuple(range(32)),
    validate_hlo=False,
    accepted_source_context=True,
)
accepted_contract = validate_integrated_dense_rms_stablehlo(
    accepted.stablehlo,
    accepted_source_context=True,
)
assert accepted_contract["passed"]
assert accepted.accepted_source_context is True
assert accepted_contract["exact_embedding_collective"] is True
accepted_report = lint_hlo(
    parse_hlo_module(accepted.optimized_hlo),
    integrated_dense_rms_hlo_policy(
        tuple(range(32)), accepted_source_context=True
    ),
)
assert accepted_report.valid, [
    item.to_dict() for item in accepted_report.violations
]
embedding = next(
    item for item in accepted_report.module.collectives
    if "accepted_source_context_embedding_collective" in (item.op_name or "")
)
attention = next(
    item for item in accepted_report.module.collectives
    if "accepted_source_context_attention_collective" in (item.op_name or "")
)
assert _validate_preceding_attention_input(
    accepted_report, embedding, parameter_index=1
)["exact_attention_parameter_index"] == 1
assert _validate_accepted_attention_input(
    accepted_report, attention, embedding
)["exact_attention_embedding_guard"] is True
dense = next(
    item for item in accepted_report.module.collectives
    if "integrated_dense_rms_strategy_nd_collective" in (item.op_name or "")
)
accepted_root = next(
    item for item in accepted_report.module.instructions
    if item.computation.startswith("ENTRY ")
    and item.raw_line.lstrip().startswith("ROOT ")
)
# CPU lowers the scalar TPU RMS schedule into reduce-windows, so it cannot
# positively satisfy the protected scheduled validator.  It can and must
# still reach that later schedule check without misclassifying the real
# convolution-fed accepted dense collective as the sealed-U16 replay mode.
try:
    _validate_exact_dense_rms_value_flow(
        accepted_report,
        dense,
        accepted_root,
        accepted_root,
        integrated_dense=True,
        require_split_output_fusion=True,
        accepted_embedding_reduction=embedding,
        accepted_attention_reduction=attention,
    )
except ValueError as exc:
    assert "exact sealed BF16 partial" not in str(exc)
else:
    raise AssertionError("CPU graph unexpectedly satisfied the TPU schedule")
for old, new, validator in (
    (
        "slice={[0:1], [0:1]}",
        "slice={[1:2], [0:1]}",
        "attention",
    ),
    (
        "direction=EQ, metadata={op_name=\"jit(integrated)/shard_map/accepted_source_context_embedding_input/eq\"",
        "direction=NE, metadata={op_name=\"jit(integrated)/shard_map/accepted_source_context_embedding_input/eq\"",
        "embedding",
    ),
    (
        "%convert.209 = f32[32,6144]{1,0} convert(%convert.210)",
        "%rogue_attention_s16 = s16[32,6144]{1,0} convert(%convert.210)\n"
        "  %convert.209 = f32[32,6144]{1,0} convert(%rogue_attention_s16)",
        "attention",
    ),
    (
        "%convert.209 = f32[32,6144]{1,0} convert(%convert.210)",
        "%convert.209 = f32[32,6144]{0,1} convert(%convert.210)",
        "attention",
    ),
):
    mutation = accepted.optimized_hlo.replace(
        old, new, 1 if validator == "attention" else -1
    )
    assert mutation != accepted.optimized_hlo
    xla_client._xla.hlo_module_from_text(mutation)
    mutated_report = lint_hlo(
        parse_hlo_module(mutation),
        integrated_dense_rms_hlo_policy(
            tuple(range(32)), accepted_source_context=True
        ),
    )
    mutated_embedding = next(
        item for item in mutated_report.module.collectives
        if "accepted_source_context_embedding_collective" in (item.op_name or "")
    )
    mutated_attention = next(
        item for item in mutated_report.module.collectives
        if "accepted_source_context_attention_collective" in (item.op_name or "")
    )
    try:
        if validator == "embedding":
            _validate_preceding_attention_input(
                mutated_report, mutated_embedding, parameter_index=1
            )
        else:
            _validate_accepted_attention_input(
                mutated_report, mutated_attention, mutated_embedding
            )
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted source {validator} mutation passed")
print(accepted_contract["exact_graph_sha256"])
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=32",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert INTEGRATED_DENSE_RMS_STABLEHLO_SHA256 in completed.stdout
    assert INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256 in completed.stdout
    assert INTEGRATED_DENSE_ORDINAL_RMS_STABLEHLO_SHA256 in completed.stdout
    assert (
        INTEGRATED_DENSE_PREDENSE_SPLIT_RMS_STABLEHLO_SHA256
        in completed.stdout
    )
    assert INTEGRATED_DENSE_ACCEPTED_SOURCE_STABLEHLO_SHA256 in completed.stdout


def test_integrated_policy_requires_the_exact_scope() -> None:
    policy = integrated_dense_rms_hlo_policy(tuple(range(32)))
    assert policy.repeated_region_patterns == (
        r"integrated_dense_rms_strategy_nd_collective",
    )
    with pytest.raises(ValueError, match="physical ids"):
        integrated_dense_rms_hlo_policy(tuple(reversed(range(32))))
    ordinal = integrated_dense_rms_hlo_policy(
        tuple(range(32)), preceding_attention_collective=True
    )
    assert ordinal.repeated_region_patterns == (
        r"integrated_dense_rms_preceding_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    )
    accepted = integrated_dense_rms_hlo_policy(
        tuple(range(32)), accepted_source_context=True
    )
    assert accepted.repeated_region_patterns == (
        r"accepted_source_context_embedding_collective",
        r"accepted_source_context_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    )
    native = integrated_dense_rms_hlo_policy(
        tuple(range(32)), native_source_context=True
    )
    assert native.repeated_region_patterns == (
        r"native_source_context_embedding_collective",
        r"native_source_context_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    )
    with pytest.raises(ValueError, match="disjoint"):
        integrated_dense_rms_hlo_policy(
            tuple(range(32)),
            accepted_source_context=True,
            native_source_context=True,
        )
    with pytest.raises(
        ValueError, match="native source StableHLO is not pinned"
    ):
        validate_integrated_dense_rms_stablehlo(
            "module awaiting protected lowering",
            native_source_context=True,
        )


def test_portable_predense_schedule_is_bound_to_fused_gate() -> None:
    from jaxlib import xla_client

    def validate(hlo: str) -> dict[str, object]:
        report = _synthetic_predense_report(hlo)
        report.raise_for_violations()
        return dict(
            _validate_split_predense_value_flow(
                report,
                next(
                    item
                    for item in report.module.instructions
                    if item.name == "%scheduled"
                ),
                next(
                    item
                    for item in report.module.instructions
                    if item.name == "%gate"
                ),
            )
        )

    hlo = _synthetic_predense_gate_hlo()
    xla_client._xla.hlo_module_from_text(hlo)
    exact = validate(hlo)
    assert exact["exact_predense_gate_fusion_ownership"] is True
    assert exact["exact_predense_scheduled_rsqrt"] is True
    mutations = (
        hlo.replace(
            "  %inverse = f32[32]{0} fusion(%scheduled),",
            "  %rogue_scalar = f32[32]{0} add(%scheduled, %scheduled)\n"
            "  %inverse = f32[32]{0} fusion(%rogue_scalar),",
            1,
        ),
        hlo.replace(
            "  %gate = f32[32,768]{1,0} convolution(%gate_lhs, %gate_weight),",
            "  %rogue_gate_lhs = bf16[32,6144]{1,0} "
            "add(%gate_lhs, %gate_lhs)\n"
            "  %gate = f32[32,768]{1,0} "
            "convolution(%rogue_gate_lhs, %gate_weight),",
            1,
        ),
        hlo.replace(
            "  %sum = f32[32,6144]{1,0} "
            "add(%attention_f32, %residual_f32)",
            "  %sum = f32[32,6144]{1,0} "
            "add(%attention_f32, %attention_f32)",
            1,
        ),
    )
    assert all(mutation != hlo for mutation in mutations)
    for mutation in mutations:
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(ValueError):
            validate(mutation)


@pytest.mark.skipif(
    not REAL_SPLIT_TPU_HLO.is_file(),
    reason="protected split integrated TPU HLO absent",
)
def test_real_split_tpu_hlo_and_row_recompute_mutations() -> None:
    from jaxlib import xla_client

    from glm_tpu.greenfield.sharding.hlo_contract import (
        lint_hlo,
        parse_hlo_module,
    )

    hlo = REAL_SPLIT_TPU_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == REAL_SPLIT_TPU_HLO_SHA256
    contract = validate_integrated_dense_rms_hlo(
        hlo,
        tuple(range(32)),
        split_layer1_rms=True,
    )
    assert contract["exact_accepted_scheduled_reduction"] is True
    assert contract["split_output_fusion_exact"] is True
    assert contract["split_recompute_exact"] is True
    report = lint_hlo(
        parse_hlo_module(hlo),
        integrated_dense_rms_hlo_policy(tuple(range(32))),
    )
    predense = _validate_split_predense_value_flow(
        report,
        next(
            item
            for item in report.module.instructions
            if item.name == "%multiply_reduce_fusion.1"
        ),
        next(
            item
            for item in report.module.instructions
            if item.name == "%conv_general_dilated.15"
        ),
    )
    assert predense["exact_predense_gate_input"] is True
    assert predense["exact_predense_scheduled_rsqrt"] is True
    replacements = (
        (
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
            "slice(%param_1.98), slice={[0:1], [0:6144]}",
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
            "slice(%param_1.98), slice={[1:2], [0:6144]}",
        ),
        (
            "%add.55 = f32[1,6144]{1,0:T(1,128)} "
            "add(%convert_element_type.145, %convert_element_type.144)",
            "%add.55 = f32[1,6144]{1,0:T(1,128)} "
            "add(%convert_element_type.145, %convert_element_type.145)",
        ),
        (
            "%param_1.98 = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} parameter(1)",
            "%param_1.98 = bf16[32,6144]{1,0} parameter(1)",
        ),
        (
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice(",
            "%slice.36 = bf16[1,6144]{1,0} slice(",
        ),
        (
            "%convert_element_type.145 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert_element_type.145 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%add.55 = f32[1,6144]{1,0:T(1,128)} add(",
            "%add.55 = f32[1,6144]{1,0} add(",
        ),
        (
            "%mul.113 = f32[1,6144]{1,0:T(1,128)} broadcast(",
            "%mul.113 = f32[1,6144]{1,0} broadcast(",
        ),
        (
            "%mul.111 = f32[1,6144]{1,0:T(1,128)} multiply(",
            "%mul.111 = f32[1,6144]{1,0} multiply(",
        ),
        (
            "%convert_element_type.141 = bf16[1,6144]{1,0:T(2,128)(2,1)} convert(",
            "%convert_element_type.141 = bf16[1,6144]{1,0} convert(",
        ),
        (
            "%convert.2 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert.2 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%mul.112 = bf16[1,6144]{1,0:T(2,128)(2,1)} broadcast(",
            "%mul.112 = bf16[1,6144]{1,0} broadcast(",
        ),
        (
            "%convert.3 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert.3 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%mul.109 = f32[1,6144]{1,0:T(1,128)} multiply(",
            "%mul.109 = f32[1,6144]{1,0} multiply(",
        ),
        (
            "%convert.4 = bf16[1,6144]{1,0:T(2,128)(2,1)} convert(",
            "%convert.4 = bf16[1,6144]{1,0} convert(",
        ),
        (
            "ROOT %bitcast_convert_type.4 = u16[1,6144]{1,0:T(2,128)(2,1)} "
            "bitcast-convert(",
            "ROOT %bitcast_convert_type.4 = u16[1,6144]{1,0} bitcast-convert(",
        ),
        (
            "ROOT %multiply_bitcast-convert_fusion = "
            "u16[1,6144]{1,0:T(2,128)(2,1)} fusion(",
            "ROOT %multiply_bitcast-convert_fusion = u16[1,6144]{1,0} fusion(",
        ),
    )
    mutations = tuple(hlo.replace(old, new, 1) for old, new in replacements)
    assert all(mutation != hlo for mutation in mutations)
    for mutation in mutations:
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(ValueError):
            validate_integrated_dense_rms_hlo(
                mutation,
                tuple(range(32)),
                split_layer1_rms=True,
            )


@pytest.mark.skipif(
    not REAL_ORDINAL_TPU_HLO.is_file(),
    reason="protected ordinal integrated TPU HLO absent",
)
def test_real_predense_value_flow_and_mutation_refusals() -> None:
    from jaxlib import xla_client

    from glm_tpu.greenfield.sharding.hlo_contract import (
        lint_hlo,
        parse_hlo_module,
    )

    hlo = REAL_ORDINAL_TPU_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == REAL_ORDINAL_TPU_HLO_SHA256

    def validate(value: str) -> dict[str, object]:
        report = lint_hlo(
            parse_hlo_module(value),
            integrated_dense_rms_hlo_policy(
                tuple(range(32)), preceding_attention_collective=True
            ),
        )
        report.raise_for_violations()
        scheduled = next(
            item
            for item in report.module.instructions
            if item.name == "%multiply_reduce_fusion.1"
        )
        gate = next(
            item
            for item in report.module.instructions
            if item.name == "%conv_general_dilated.15"
        )
        attention = next(
            item
            for item in report.module.collectives
            if "integrated_dense_rms_preceding_attention_collective"
            in (item.op_name or "").split("/")
        )
        return dict(
            _validate_split_predense_value_flow(
                report,
                scheduled,
                gate,
                preceding_attention_reduction=attention,
            )
        )

    exact = validate(hlo)
    assert exact["exact_predense_scheduled_reduction"] is True
    assert exact["exact_predense_scheduled_rsqrt"] is True
    assert exact["exact_predense_recompute"] is True
    assert exact["exact_predense_gate_fusion_ownership"] is True
    mutations = (
        hlo.replace(
            "  %add_rsqrt_fusion.1 = f32[32]{0:T(128)S(3)} "
            "fusion(%get-tuple-element),",
            "  %rogue_predense_scalar = f32[32]{0:T(128)S(3)} "
            "add(%get-tuple-element, %get-tuple-element)\n"
            "  %add_rsqrt_fusion.1 = f32[32]{0:T(128)S(3)} "
            "fusion(%rogue_predense_scalar),",
            1,
        ),
        hlo.replace(
            "  %conv_general_dilated.15 = f32[32,768]{1,0:T(8,128)} "
            "convolution(%fusion.23, %fusion.22),",
            "  %rogue_predense_gate = bf16[32,6144]"
            "{1,0:T(8,128)(2,1)} add(%fusion.23, %fusion.23)\n"
            "  %conv_general_dilated.15 = f32[32,768]{1,0:T(8,128)} "
            "convolution(%rogue_predense_gate, %fusion.22),",
            1,
        ),
        hlo.replace(
            "  %add.48 = f32[32,6144]{1,0:T(8,128)} "
            "add(%convert_element_type.132, %convert_element_type.131),",
            "  %add.48 = f32[32,6144]{1,0:T(8,128)} "
            "add(%convert_element_type.132, %convert_element_type.132),",
            1,
        ),
    )
    assert all(mutation != hlo for mutation in mutations)
    for mutation in mutations:
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(ValueError):
            validate(mutation)


@pytest.mark.skipif(
    not REAL_ACCEPTED_SOURCE_TPU_HLO.is_file(),
    reason="protected accepted-source integrated TPU HLO absent",
)
def test_real_accepted_source_tpu_hlo_and_mutation_refusals() -> None:
    from jaxlib import xla_client

    hlo = REAL_ACCEPTED_SOURCE_TPU_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == (
        REAL_ACCEPTED_SOURCE_TPU_HLO_SHA256
    )

    def validate(value: str) -> dict[str, object]:
        return validate_integrated_dense_rms_hlo(
            value,
            tuple(range(32)),
            accepted_source_context=True,
        )

    exact = validate(hlo)
    assert exact["passed"] is True
    assert exact["accepted_source_context"] is True
    assert exact["exact_accepted_attention_input"] is True
    assert exact["exact_attention_embedding_guard"] is True
    assert exact["exact_predense_gate_input"] is True
    assert exact["exact_reduction_operand_graph"] is True
    assert exact["exact_weighted_operand_graph"] is True
    assert exact["exact_result_binding"] is True
    assert exact["residual_source_mode"] == (
        "accepted_embedding_predicate_plus_attention"
    )
    contraction = exact["contraction"]
    assert isinstance(contraction, dict)
    assert contraction["exact_packed_weight_lineage"] is True
    assert contraction["exact_accepted_weight_layout"] is True

    replacements = (
        (
            "%is_finite.6 = pred[1,1]{1,0:T(4,128)(4,1)} is-finite(",
            "%is_finite.6 = pred[1,1]{1,0:T(8,128)(4,1)} is-finite(",
        ),
        (
            "%select_n.42 = bf16[32,6144]{1,0:T(8,128)(2,1)} "
            "select(%broadcast_in_dim.45, %param_1.120, %broadcast.38)",
            "%select_n.42 = bf16[32,6144]{1,0:T(8,128)(2,1)} "
            "select(%broadcast_in_dim.45, %param_0.131, %broadcast.38)",
        ),
        (
            "%select_n.40 = bf16[32,6144]{1,0:T(8,128)(2,1)} "
            "select(%broadcast_in_dim.43, %param_2.83, %broadcast.36)",
            "%select_n.40 = bf16[32,6144]{1,0:T(8,128)(2,1)} "
            "select(%broadcast_in_dim.43, %param_1.119, %broadcast.36)",
        ),
        (
            "%convert_element_type.135 = "
            "bf16[32,6144]{1,0:T(8,128)(2,1)} convert(",
            "%convert_element_type.135 = "
            "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} convert(",
        ),
        (
            "ROOT %convert.2 = "
            "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} convert(",
            "ROOT %convert.2 = "
            "bf16[32,6144]{1,0:T(8,128)(2,1)} convert(",
        ),
    )
    mutations = [hlo.replace(old, new, 1) for old, new in replacements]
    mutations.append(
        hlo.replace(
            "%param.12 = bf16[6144]{0:T(1024)(128)(2,1)} "
            "parameter(3)",
            "%param.12 = bf16[6144]{0:T(1024)(128)(2,1)} "
            "parameter(4)",
            1,
        ).replace(
            "%param.13 = f8e4m3fn[1,1,6144,768]"
            "{3,2,1,0:T(32,128)(4,1)} parameter(4)",
            "%param.13 = f8e4m3fn[1,1,6144,768]"
            "{3,2,1,0:T(32,128)(4,1)} parameter(3)",
            1,
        )
    )
    assert all(mutation != hlo for mutation in mutations)
    for mutation in mutations:
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(ValueError):
            validate(mutation)


def test_checkpoint_success_pin_and_missing_mutated_refusals(
    tmp_path: Path,
) -> None:
    marker = REAL_STAGE0_SLOT0.parents[2] / "SUCCESS"
    if not marker.is_file():
        pytest.skip("protected checkpoint marker absent")
    assert __import__("hashlib").sha256(marker.read_bytes()).hexdigest() == (
        CHECKPOINT_SUCCESS_SHA256
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="checkpoint SUCCESS drifted"):
        validate_integrated_checkpoint_success(checkpoint)
    shutil.copy2(marker, checkpoint / "SUCCESS")
    assert validate_integrated_checkpoint_success(checkpoint) == (
        CHECKPOINT_SUCCESS_SHA256
    )
    (checkpoint / "SUCCESS").write_bytes(marker.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="checkpoint SUCCESS drifted"):
        validate_integrated_checkpoint_success(checkpoint)


@pytest.mark.skipif(
    not REAL_LEGACY_INTEGRATED_RUN.is_dir() or not REAL_CHECKPOINT_ROOT.is_dir(),
    reason="sealed legacy integrated run/checkpoint absent",
)
def test_genuine_legacy_integrated_archive_still_revalidates() -> None:
    summary = validate_strategy_nd_integrated_dense_rms(
        REAL_LEGACY_INTEGRATED_RUN,
        checkpoint_root=REAL_CHECKPOINT_ROOT,
        expected_code_hash="d7872b582181e8c2518da2d8785b109a733e92b1",
        expected_run_tag=REAL_LEGACY_INTEGRATED_RUN.name,
    )
    assert summary["classification"] == (
        "integrated_dense_rms_matches_db548_control"
    )
    assert summary["mismatch_count"] == 1


def test_integrated_comparison_and_capture_are_recomputed_from_arrays() -> None:
    expected = np.arange(6144, dtype=np.uint16)
    exact = _recompute_comparison(expected.copy(), expected)
    assert exact["classification"] == "integrated_dense_rms_exact_accepted"
    assert exact["mismatch_count"] == 0
    split_exact = _recompute_comparison(
        expected.copy(), expected, split_layer1_rms=True
    )
    assert split_exact["classification"] == (
        "integrated_dense_split_rms_exact_accepted"
    )
    ordinal_exact = _recompute_comparison(
        expected.copy(),
        expected,
        split_layer1_rms=True,
        preceding_attention_collective=True,
    )
    assert ordinal_exact["classification"] == (
        "integrated_dense_ordinal_rms_exact_accepted"
    )
    predense_exact = _recompute_comparison(
        expected.copy(),
        expected,
        split_layer1_rms=True,
        split_predense_rms=True,
    )
    assert predense_exact["classification"] == (
        "integrated_dense_predense_split_rms_exact_accepted"
    )
    accepted_source_exact = _recompute_comparison(
        expected.copy(), expected, accepted_source_context=True
    )
    assert accepted_source_exact["classification"] == (
        "integrated_dense_accepted_source_context_exact_accepted"
    )
    native_source_exact = _recompute_comparison(
        expected.copy(), expected, native_source_context=True
    )
    assert native_source_exact["classification"] == (
        "integrated_dense_native_source_context_exact_accepted"
    )
    digest = array_sha256(expected)
    capture = {
        "invocation_count": 2,
        "local_replica_output_sha256": [digest] * 4,
        "output_bits_sha256": digest,
        "repeated_local_replica_output_sha256": [digest] * 4,
        "repeated_output_bits_sha256": digest,
    }
    _validate_capture(capture, expected)
    capture["invocation_count"] = True
    with pytest.raises(ValueError, match="deterministic capture"):
        _validate_capture(capture, expected)


def test_hlo_prevalidation_record_is_exact_and_non_promoting() -> None:
    legacy = {
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "split_layer1_rms": True,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        legacy,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=True,
    )
    record = {
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "preceding_attention_collective": True,
        "split_layer1_rms": True,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        record,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=True,
        preceding_attention_collective=True,
    )
    predense_record = {
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "split_layer1_rms": True,
        "split_predense_rms": True,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        predense_record,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=True,
        split_predense_rms=True,
    )
    accepted_record = {
        "accepted_source_context": True,
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "split_layer1_rms": False,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        accepted_record,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=False,
        accepted_source_context=True,
    )
    native_record = {
        "native_source_context": True,
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "split_layer1_rms": False,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        native_record,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=False,
        native_source_context=True,
    )
    for key, value in (
        ("validated", True),
        ("performance_claim", True),
        ("preceding_attention_collective", False),
        ("split_layer1_rms", False),
        ("optimized_hlo_sha256", "3" * 64),
    ):
        mutation = dict(record)
        mutation[key] = value
        with pytest.raises(ValueError, match="prevalidation record drifted"):
            _validate_hlo_prevalidation(
                mutation,
                optimized_hlo_sha256="1" * 64,
                stablehlo_sha256="2" * 64,
                split_layer1_rms=True,
                preceding_attention_collective=True,
            )


def test_protected_integrated_wrapper_is_default_off_and_success_last() -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    assert "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_RMS_REPLAY:-0" in wrapper
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_SPLIT_RMS_REPLAY:-0"
        in wrapper
    )
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_ORDINAL_RMS_REPLAY:-0"
        in wrapper
    )
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_PREDENSE_SPLIT_RMS_REPLAY:-0"
        in wrapper
    )
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_ACCEPTED_SOURCE_CONTEXT_REPLAY:-0"
        in wrapper
    )
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_NATIVE_SOURCE_CONTEXT_REPLAY:-0"
        in wrapper
    )
    assert "--integrated-accepted-source-context" in wrapper
    assert "--integrated-native-source-context" in wrapper
    assert "--native-attention-input" in wrapper
    assert "--integrated-preceding-attention-collective" in wrapper
    assert "--integrated-split-predense-rms" in wrapper
    assert "--integrated-split-layer1-rms" in wrapper
    assert "--mode strategy_nd_integrated_dense_rms" in wrapper
    assert "validate_strategy_nd_integrated_dense_rms" in wrapper
    assert '"$RMS_REPLAY" "$INTEGRATED_REPLAY"' in wrapper
    assert wrapper.index("strict_census post") < wrapper.index(
        '"$REMOTE_PREFIX/SUCCESS" >/dev/null'
    )
    runner = (
        REPO / "scripts/greenfield/microbench_collectives.py"
    ).read_text()
    start = runner.index("def _run_strategy_nd_integrated_dense_rms(")
    end = runner.index("\ndef _run_strategy_nd_fingerprint(", start)
    integrated = runner[start:end]
    assert "validate_hlo=False" in integrated
    assert "hlo_prevalidation.json" in integrated
    persisted = integrated.index(
        'f"greenfield-integrated-hlo-persisted-{label}"'
    )
    stable_validator = integrated.index(
        "stablehlo_contract = validate_integrated_dense_rms_stablehlo("
    )
    assert integrated.index("hlo_prevalidation.json") < persisted
    assert persisted < stable_validator
    assert integrated.index("hlo_prevalidation.json") < integrated.index(
        "execute_integrated_dense_rms("
    )
    assert integrated.index("hlo_prevalidation.json") < integrated.index(
        "execute_native_source_context("
    )
