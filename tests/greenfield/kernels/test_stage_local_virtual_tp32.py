from __future__ import annotations

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import glm_tpu.greenfield.kernels.stage_local as stage_local
from glm_tpu.greenfield.kernels.stage_local import (
    _decode_dense_fp8_in_out,
    _dense_bf16_convolution,
    _strategy_nd_row0_bf16_reduce,
    _sum_virtual_dcp_bf16_partials,
    _virtual_dense_convolution_down_partials,
    _virtual_dense_down_partials,
)


def _bf16_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(
        left.astype(np.float32) + right.astype(np.float32),
        dtype=jnp.bfloat16,
    )


def test_virtual_dcp_bf16_trees_are_explicit_and_numerically_distinct() -> None:
    source = np.asarray(
        [
            14080.0,
            -1.9140625,
            -2.875,
            2496.0,
            -2496.0,
            -0.86328125,
            0.00927734375,
            540.0,
        ],
        dtype=jnp.bfloat16,
    ).reshape(8, 1, 1)

    sequential = source[0]
    for value in source[1:]:
        sequential = _bf16_add(sequential, value)
    pairwise = tuple(source[index] for index in range(8))
    while len(pairwise) > 1:
        pairwise = tuple(
            _bf16_add(pairwise[index], pairwise[index + 1])
            for index in range(0, len(pairwise), 2)
        )

    actual_sequential = np.asarray(
        jax.jit(
            lambda value: _sum_virtual_dcp_bf16_partials(
                value, pairwise=False
            )
        )(jnp.asarray(source))
    )
    actual_pairwise = np.asarray(
        jax.jit(
            lambda value: _sum_virtual_dcp_bf16_partials(
                value, pairwise=True
            )
        )(jnp.asarray(source))
    )
    np.testing.assert_array_equal(actual_sequential, sequential)
    np.testing.assert_array_equal(actual_pairwise, pairwise[0])
    assert not np.array_equal(
        actual_sequential.view(np.uint16),
        actual_pairwise.view(np.uint16),
    )

    sequential_hlo = jax.jit(
        lambda value: _sum_virtual_dcp_bf16_partials(
            value, pairwise=False
        )
    ).lower(jnp.asarray(source)).as_text()
    pairwise_hlo = jax.jit(
        lambda value: _sum_virtual_dcp_bf16_partials(
            value, pairwise=True
        )
    ).lower(jnp.asarray(source)).as_text()
    assert sequential_hlo.count("stablehlo.optimization_barrier") == 7
    assert pairwise_hlo.count("stablehlo.optimization_barrier") == 7
    assert sequential_hlo != pairwise_hlo


def test_virtual_dcp_bf16_sum_rejects_unrounded_or_wrong_shard_count() -> None:
    with np.testing.assert_raises_regex(ValueError, "eight rank-two"):
        _sum_virtual_dcp_bf16_partials(
            jnp.zeros((7, 1, 4), dtype=jnp.bfloat16), pairwise=False
        )
    with np.testing.assert_raises_regex(ValueError, "rounded BF16"):
        _sum_virtual_dcp_bf16_partials(
            jnp.zeros((8, 1, 4), dtype=jnp.float32), pairwise=True
        )


def _strategy_nd_row0_numpy(
    model_partials: np.ndarray,
    *,
    middle_y_cross: bool = True,
    alternate_z: bool = True,
) -> np.ndarray:
    model_to_physical = np.asarray(
        (
            0, 8, 16, 24, 2, 10, 18, 26,
            4, 12, 20, 28, 6, 14, 22, 30,
            1, 9, 17, 25, 3, 11, 19, 27,
            5, 13, 21, 29, 7, 15, 23, 31,
        ),
        dtype=np.int32,
    )
    physical = np.empty_like(model_partials)
    physical[model_to_physical] = model_partials
    values = physical.reshape(4, 4, 2, 1, 6144).transpose(1, 2, 0, 3, 4)

    def four(source: np.ndarray, cross: bool) -> np.ndarray:
        if cross:
            return _bf16_add(
                _bf16_add(source[0], source[3]),
                _bf16_add(source[1], source[2]),
            )
        return _bf16_add(
            _bf16_add(source[0], source[1]),
            _bf16_add(source[2], source[3]),
        )

    y = np.concatenate(
        (
            four(values[..., :2048], False),
            four(values[..., 2048:4096], middle_y_cross),
            four(values[..., 4096:], False),
        ),
        axis=-1,
    )
    x = _bf16_add(y[0], y[1])
    return np.concatenate(
        tuple(
            four(
                x[..., start : start + 256],
                bool((start // 256) % 2) if alternate_z else False,
            )
            for start in range(0, 6144, 256)
        ),
        axis=-1,
    )


def test_strategy_nd_row0_replays_db533_physical_schedule() -> None:
    rng = np.random.default_rng(533)
    physical = np.asarray(
        rng.standard_normal((32, 1, 6144), dtype=np.float32)
        * np.exp2(rng.integers(-4, 9, size=(32, 1, 6144))),
        dtype=jnp.bfloat16,
    )
    model_to_physical = np.asarray(
        (
            0, 8, 16, 24, 2, 10, 18, 26,
            4, 12, 20, 28, 6, 14, 22, 30,
            1, 9, 17, 25, 3, 11, 19, 27,
            5, 13, 21, 29, 7, 15, 23, 31,
        )
    )
    model = physical[model_to_physical]
    expected = _strategy_nd_row0_numpy(model)
    actual = np.asarray(
        jax.jit(_strategy_nd_row0_bf16_reduce)(jnp.asarray(model))
    )
    np.testing.assert_array_equal(
        actual.view(np.uint16), expected.view(np.uint16)
    )

    wrong_mapping = model.copy()
    wrong_mapping[[0, 1]] = wrong_mapping[[1, 0]]
    mutants = (
        _strategy_nd_row0_numpy(wrong_mapping),
        _strategy_nd_row0_numpy(model, middle_y_cross=False),
        _strategy_nd_row0_numpy(model, alternate_z=False),
        np.asarray(
            model.astype(np.float32).sum(axis=0),
            dtype=jnp.bfloat16,
        ),
    )
    assert all(
        np.count_nonzero(mutant.view(np.uint16) != expected.view(np.uint16))
        for mutant in mutants
    )

    hlo = jax.jit(_strategy_nd_row0_bf16_reduce).lower(
        jnp.asarray(model)
    ).as_text()
    assert hlo.count("stablehlo.optimization_barrier") >= 82


def test_strategy_nd_row0_rejects_wrong_geometry_or_dtype() -> None:
    with np.testing.assert_raises_regex(ValueError, "32 model partials"):
        _strategy_nd_row0_bf16_reduce(
            jnp.zeros((31, 1, 6144), dtype=jnp.bfloat16)
        )
    with np.testing.assert_raises_regex(ValueError, "rounded BF16"):
        _strategy_nd_row0_bf16_reduce(
            jnp.zeros((32, 1, 6144), dtype=jnp.float32)
        )


def test_dense_convolution_primitives_preserve_fp8_decode_and_f32_accumulation() -> None:
    bits = jnp.asarray(
        np.arange(16, dtype=np.uint8).reshape(4, 4), dtype=jnp.uint8
    )
    scale = jnp.asarray([[0.5]], dtype=jnp.float32)
    decoded = _decode_dense_fp8_in_out(
        bits, scale, block_shape=(4, 4)
    )
    expected = jax.lax.bitcast_convert_type(
        bits, jnp.float8_e4m3fn
    ).astype(jnp.float32) * jnp.float32(0.5)
    np.testing.assert_array_equal(
        np.asarray(decoded).view(np.uint16),
        np.asarray(expected.astype(jnp.bfloat16)).view(np.uint16),
    )

    lhs = jnp.asarray([[1.0, -2.0, 0.5, 3.0]], dtype=jnp.bfloat16)
    actual = _dense_bf16_convolution(lhs, decoded)
    reference = jax.lax.dot_general(
        lhs,
        decoded,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(reference).view(np.uint16),
    )


def test_virtual_dense_convolution_has_one_row_and_sixteen_convolutions() -> None:
    shapes = (
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((3072, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((24, 48), jnp.float32),
        jax.ShapeDtypeStruct((3072, 6144), jnp.uint8),
        jax.ShapeDtypeStruct((24, 48), jnp.float32),
        jax.ShapeDtypeStruct((6144, 3072), jnp.uint8),
        jax.ShapeDtypeStruct((48, 24), jnp.float32),
    )
    lowered = jax.jit(
        lambda *args: _virtual_dense_convolution_down_partials(
            *args, block_shape=(128, 128)
        )
    ).lower(*shapes)
    assert lowered.out_info.shape == (8, 1, 6144)
    assert lowered.out_info.dtype == jnp.bfloat16
    stablehlo = lowered.as_text()
    assert stablehlo.count("stablehlo.convolution") == 16
    assert "tensor<1x6144xbf16>" in stablehlo
    assert "tensor<32x6144xbf16>" not in stablehlo


def test_virtual_dense_convolution_rejects_nonexact_contract() -> None:
    good = (
        jnp.zeros((1, 6144), dtype=jnp.bfloat16),
        jnp.zeros((3072, 6144), dtype=jnp.uint8),
        jnp.ones((24, 48), dtype=jnp.float32),
        jnp.zeros((3072, 6144), dtype=jnp.uint8),
        jnp.ones((24, 48), dtype=jnp.float32),
        jnp.zeros((6144, 3072), dtype=jnp.uint8),
        jnp.ones((48, 24), dtype=jnp.float32),
    )
    with np.testing.assert_raises_regex(ValueError, "requires 128x128"):
        _virtual_dense_convolution_down_partials(
            *good, block_shape=(64, 128)
        )
    with np.testing.assert_raises_regex(ValueError, "input must be BF16"):
        _virtual_dense_convolution_down_partials(
            good[0].astype(jnp.float32), *good[1:], block_shape=(128, 128)
        )


def test_virtual_dense_pallas_partials_accept_exact_pp16_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Shaped:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

        def __getitem__(self, key: object) -> "Shaped":
            return self

    calls = []

    def fake_fused(*args: object, **kwargs: object) -> jax.Array:
        calls.append((args, kwargs))
        return jnp.zeros((1, 6144), dtype=jnp.bfloat16)

    monkeypatch.setattr(stage_local, "fp8_fused_block_swiglu", fake_fused)
    result = _virtual_dense_down_partials(
        Shaped((1, 6144)),
        Shaped((6144, 6144)),
        Shaped((48, 48)),
        Shaped((6144, 6144)),
        Shaped((48, 48)),
        Shaped((6144, 6144)),
        Shaped((48, 48)),
        block_shape=(128, 128),
        linear_interpret=False,
    )
    assert result.shape == (16, 1, 6144)
    assert result.dtype == jnp.bfloat16
    assert len(calls) == 16


def test_virtual_attention_partials_accept_exact_pp16_feature_halves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Shaped:
        def __init__(
            self,
            shape: tuple[int, ...],
            dtype: object,
            *,
            name: str,
            key: object | None = None,
        ):
            self.shape = shape
            self.dtype = dtype
            self.name = name
            self.key = key

        def __getitem__(self, key: object) -> "Shaped":
            if not isinstance(key, tuple) or len(key) != 2:
                raise AssertionError(f"unexpected {self.name} slice: {key!r}")
            row_key, column_key = key
            if row_key != slice(None):
                row_count = row_key.stop - row_key.start
            else:
                row_count = self.shape[0]
            column_count = column_key.stop - column_key.start
            return Shaped(
                (row_count, column_count),
                self.dtype,
                name=self.name,
                key=key,
            )

    calls = []

    def fake_projection(*args: object, **kwargs: object) -> jax.Array:
        calls.append((args, kwargs))
        return jnp.zeros((1, 3072), dtype=jnp.bfloat16)

    monkeypatch.setattr(
        stage_local, "fp8_strategy_nd_attention_matmul", fake_projection
    )
    output = Shaped((1, 8192), jnp.bfloat16, name="output")
    bits = Shaped((6144, 8192), jnp.uint8, name="bits")
    scale = Shaped((48, 64), jnp.float32, name="scale")
    for feature_half in (0, 1):
        result = stage_local._virtual_attention_output_feature_half_partials(
            output,
            bits,
            scale,
            feature_half=feature_half,
            block_shape=(128, 128),
            linear_interpret=False,
        )
        assert result.shape == (16, 1, 3072)
        assert result.dtype == jnp.bfloat16
    assert len(calls) == 32
    for feature_half in (0, 1):
        for shard in range(16):
            args, _ = calls[feature_half * 16 + shard]
            lhs, weight, weight_scale = args[:3]
            assert lhs.name == "output"
            assert lhs.key == (
                slice(None),
                slice(shard * 512, (shard + 1) * 512),
            )
            assert weight.name == "bits"
            assert weight.key == (
                slice(feature_half * 3072, (feature_half + 1) * 3072),
                slice(shard * 512, (shard + 1) * 512),
            )
            assert weight_scale.name == "scale"
            assert weight_scale.key == (
                slice(feature_half * 24, (feature_half + 1) * 24),
                slice(shard * 4, (shard + 1) * 4),
            )

    with pytest.raises(ValueError, match="weight dtypes"):
        stage_local._virtual_attention_output_feature_half_partials(
            output,
            Shaped(
                (6144, 8192),
                jnp.float8_e4m3fn,
                name="bits",
            ),
            scale,
            feature_half=0,
            block_shape=(128, 128),
            linear_interpret=False,
        )


def test_virtual_attention_full_width_leaves_are_immediately_sliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Shaped:
        def __init__(
            self,
            shape: tuple[int, ...],
            dtype: object,
            *,
            name: str,
            key: object | None = None,
        ):
            self.shape = shape
            self.dtype = dtype
            self.name = name
            self.key = key

        def __getitem__(self, key: object) -> "Shaped":
            if not isinstance(key, tuple) or len(key) != 2:
                raise AssertionError(f"unexpected {self.name} slice: {key!r}")
            row_key, column_key = key
            row_count = (
                self.shape[0]
                if row_key == slice(None)
                else row_key.stop - row_key.start
            )
            return Shaped(
                (row_count, column_key.stop - column_key.start),
                self.dtype,
                name=self.name,
                key=key,
            )

    calls = []
    rounded = jnp.arange(6144, dtype=jnp.float32)[None, :].astype(
        jnp.bfloat16
    )

    def fake_projection(*args: object, **kwargs: object) -> jax.Array:
        calls.append((args, kwargs))
        return rounded

    monkeypatch.setattr(
        stage_local, "fp8_strategy_nd_attention_matmul", fake_projection
    )
    output = Shaped((1, 8192), jnp.bfloat16, name="output")
    bits = Shaped((6144, 8192), jnp.uint8, name="bits")
    scale = Shaped((48, 64), jnp.float32, name="scale")
    half0, half1 = (
        stage_local._virtual_attention_output_full_width_then_slice_partials(
            output,
            bits,
            scale,
            block_shape=(128, 128),
            linear_interpret=False,
        )
    )
    assert half0.shape == half1.shape == (16, 1, 3072)
    np.testing.assert_array_equal(np.asarray(half0[0]), np.asarray(rounded[:, :3072]))
    np.testing.assert_array_equal(np.asarray(half1[0]), np.asarray(rounded[:, 3072:]))
    assert len(calls) == 16
    for shard, (args, _) in enumerate(calls):
        lhs, weight, weight_scale = args[:3]
        assert lhs.key == (
            slice(None),
            slice(shard * 512, (shard + 1) * 512),
        )
        assert weight.key == (
            slice(None),
            slice(shard * 512, (shard + 1) * 512),
        )
        assert weight_scale.key == (
            slice(None),
            slice(shard * 4, (shard + 1) * 4),
        )


def test_pp16_pregathered_attention_is_two_exact_h16_b512_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = stage_local.MlaNumericalContract()
    q_nope = jnp.zeros((1, 32, 512), dtype=jnp.bfloat16)
    q_rope = jnp.zeros((1, 32, 64), dtype=jnp.bfloat16)
    selected_cache = jnp.zeros((1, 2048, 640), dtype=jnp.bfloat16)
    valid_counts = jnp.asarray([2048], dtype=jnp.int32)
    calls = []

    def fake_attention(*args: object, **kwargs: object) -> jax.Array:
        calls.append((args, kwargs))
        return jnp.full(
            (1, 16, 512), len(calls), dtype=jnp.bfloat16
        )

    monkeypatch.setattr(
        stage_local, "pregathered_sparse_mla_pallas", fake_attention
    )
    result = stage_local._pregathered_b512_attention_lp2_h16_pair(
        q_nope,
        q_rope,
        selected_cache,
        valid_counts,
        contract=contract,
        interpret=False,
    )
    assert result.shape == (1, 32, 512)
    assert np.all(np.asarray(result[:, :16], dtype=np.float32) == 1)
    assert np.all(np.asarray(result[:, 16:], dtype=np.float32) == 2)
    assert len(calls) == 2
    for head_half, (args, kwargs) in enumerate(calls):
        np.testing.assert_array_equal(
            np.asarray(args[0]),
            np.asarray(q_nope[:, head_half * 16 : (head_half + 1) * 16]),
        )
        np.testing.assert_array_equal(
            np.asarray(args[1]),
            np.asarray(q_rope[:, head_half * 16 : (head_half + 1) * 16]),
        )
        assert args[2] is selected_cache
        assert args[3] is valid_counts
        assert kwargs["contract"].num_heads == 16
        assert kwargs["contract"].top_k == 2048
        assert kwargs["config"].segment_block == 512

    with pytest.raises(ValueError, match="exact GLM MLA contract"):
        stage_local._pregathered_b512_attention_lp2_h16_pair(
            q_nope,
            q_rope,
            selected_cache,
            valid_counts,
            contract=stage_local.MlaNumericalContract(num_heads=32),
            interpret=False,
        )


def test_pp16_pregathered_attention_jaxpr_has_exact_h16_b512_lineage() -> None:
    contract = stage_local.MlaNumericalContract()
    arguments = (
        jax.ShapeDtypeStruct((1, 32, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 32, 64), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 2048, 640), jnp.bfloat16),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )

    def execute(*args: object) -> jax.Array:
        return stage_local._pregathered_b512_attention_lp2_h16_pair(
            *args,
            contract=contract,
            interpret=False,
        )

    jaxpr = str(jax.make_jaxpr(execute)(*arguments))
    assert jaxpr.count(
        "name=greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"
    ) == 2
    assert "h32_k2048" not in jaxpr
    assert "all_gather" not in jaxpr
    assert "bf16[1,32,512] = concatenate[dimension=1]" in jaxpr


def test_pp16_feature_attention_contract_rejects_every_legacy_escape() -> None:
    valid = {
        "axis_name": "feature",
        "groups": ((0, 1),),
        "feature_pairs": ((0, 1), (1, 0)),
        "cache_layout": stage_local.StageLocalKvLayout(
            logical_page_size=512,
            local_parallel_size=2,
            packed_cache_width=640,
        ),
        "contract": stage_local.MlaNumericalContract(),
        "pregathered_b512_attention": True,
        "linear_backend": "pallas",
        "add_residual": False,
        "reconstruct_output_fp32": False,
        "virtual_tp32_reduction_association": None,
        "replicated_monolithic_attention": False,
        "capture_ingredients": False,
    }
    stage_local._require_pp16_feature_sharded_attention_contract(**valid)
    mutations = (
        {"axis_name": ""},
        {"groups": None},
        {"groups": ((0, 1, 2, 3),)},
        {"feature_pairs": ()},
        {
            "cache_layout": stage_local.StageLocalKvLayout(
                logical_page_size=512,
                local_parallel_size=4,
                packed_cache_width=640,
            )
        },
        {"contract": stage_local.MlaNumericalContract(num_heads=32)},
        {"pregathered_b512_attention": False},
        {"linear_backend": "reference"},
        {"add_residual": True},
        {"reconstruct_output_fp32": True},
        {"virtual_tp32_reduction_association": "strategy_nd_row0_bf16"},
        {"replicated_monolithic_attention": True},
        {"capture_ingredients": True},
    )
    for mutation in mutations:
        candidate = dict(valid)
        candidate.update(mutation)
        with pytest.raises(ValueError, match="isolated PP16"):
            stage_local._require_pp16_feature_sharded_attention_contract(
                **candidate
            )


def test_strategy_nd_row0_forced_four_device_gather_is_local_and_exact() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    _reduce_strategy_nd_row0_bf16_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("stage",))
source = np.arange(4 * 8 * 6144, dtype=np.float32).reshape(4, 8, 1, 6144)
source = jnp.asarray(np.sin(source / 37.0), dtype=jnp.bfloat16)
sharded = jax.device_put(source, NamedSharding(mesh, P("stage", None, None, None)))

def mapped(local):
    reduced = _reduce_strategy_nd_row0_bf16_partials(
        local[0], axis_name="stage", groups=((0, 1, 2, 3),)
    )
    return reduced[None, ...]

execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=P("stage", None, None, None),
    out_specs=P("stage", None, None),
    check_vma=False,
)
lowered = jax.jit(execute).lower(sharded)
actual = np.asarray(jax.jit(execute)(sharded)).view(np.uint16)
print(json.dumps({
    "all_gather_count": lowered.as_text().count("stablehlo.all_gather"),
    "lane_replication": bool(np.all(actual == actual[0])),
    "shape": list(actual.shape),
}))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "all_gather_count": 1,
        "lane_replication": True,
        "shape": [4, 1, 6144],
    }


def test_strategy_nd_row0_forced_two_device_y_x_z_is_local_and_exact() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    _reduce_strategy_nd_row0_bf16_partials,
    _strategy_nd_row0_bf16_reduce,
)

mesh = Mesh(np.asarray(jax.devices()), ("stage",))
source = np.arange(2 * 16 * 6144, dtype=np.float32).reshape(2, 16, 1, 6144)
source = jnp.asarray(np.sin(source / 37.0), dtype=jnp.bfloat16)
sharded = jax.device_put(source, NamedSharding(mesh, P("stage", None, None, None)))

def mapped(local):
    reduced = _reduce_strategy_nd_row0_bf16_partials(
        local[0], axis_name="stage", groups=((0, 1),)
    )
    return reduced[None, ...]

execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=P("stage", None, None, None),
    out_specs=P("stage", None, None),
    check_vma=False,
)
lowered = jax.jit(execute).lower(sharded)
actual = np.asarray(jax.jit(execute)(sharded))
expected = np.asarray(
    jax.jit(_strategy_nd_row0_bf16_reduce)(source.reshape(32, 1, 6144))
)
print(json.dumps({
    "all_gather_count": lowered.as_text().count("stablehlo.all_gather"),
    "all_reduce_count": lowered.as_text().count("stablehlo.all_reduce"),
    "exact": bool(np.all(actual[0].view(np.uint16) == expected.view(np.uint16))),
    "lane_replication": bool(np.all(actual == actual[0])),
    "payload_present": "tensor<4x1x6144xbf16>" in lowered.as_text(),
    "shape": list(actual.shape),
}))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "all_gather_count": 0,
        "all_reduce_count": 1,
        "exact": True,
        "lane_replication": True,
        "payload_present": True,
        "shape": [2, 1, 6144],
    }


def test_strategy_nd_feature_half_forced_two_device_is_exact_and_never_full() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    _reduce_strategy_nd_feature_half_bf16_partials,
    _strategy_nd_row0_bf16_reduce,
)

mesh = Mesh(np.asarray(jax.devices()), ("stage",))
source = np.arange(2 * 16 * 6144, dtype=np.float32).reshape(2, 16, 1, 6144)
source = jnp.asarray(np.sin(source / 37.0), dtype=jnp.bfloat16)
half0 = jax.device_put(
    source[..., :3072],
    NamedSharding(mesh, P("stage", None, None, None)),
)
half1 = jax.device_put(
    source[..., 3072:],
    NamedSharding(mesh, P("stage", None, None, None)),
)

def mapped(local_half0, local_half1):
    reduced = _reduce_strategy_nd_feature_half_bf16_partials(
        local_half0[0],
        local_half1[0],
        axis_name="stage",
        pairs=((0, 1), (1, 0)),
    )
    return reduced[None, ...]

execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(
        P("stage", None, None, None),
        P("stage", None, None, None),
    ),
    out_specs=P("stage", None, None),
    check_vma=False,
)
lowered = jax.jit(execute).lower(half0, half1)
actual = np.asarray(jax.jit(execute)(half0, half1))
expected = np.asarray(
    jax.jit(_strategy_nd_row0_bf16_reduce)(source.reshape(32, 1, 6144))
)
concatenated = np.concatenate((actual[0], actual[1]), axis=-1)
stablehlo = lowered.as_text()
print(json.dumps({
    "all_gather_count": stablehlo.count("stablehlo.all_gather"),
    "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
    "collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
    "exact": bool(np.array_equal(
        concatenated.view(np.uint16), expected.view(np.uint16)
    )),
    "forbidden_full_hidden": "tensor<1x6144xbf16>" in stablehlo,
    "payload_present": "tensor<4x1x3072xbf16>" in stablehlo,
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
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "all_gather_count": 0,
        "all_reduce_count": 0,
        "collective_permute_count": 1,
        "exact": True,
        "forbidden_full_hidden": False,
        "payload_present": True,
        "shape": [2, 1, 3072],
    }


def test_strategy_nd_feature_half_rejects_wrong_geometry_dtype_or_pairs() -> None:
    from glm_tpu.greenfield.kernels.stage_local import (
        _reduce_strategy_nd_feature_half_bf16_partials,
        _strategy_nd_feature_half_y_reduce,
    )

    valid = jnp.zeros((16, 1, 3072), dtype=jnp.bfloat16)
    with np.testing.assert_raises_regex(ValueError, "16 one-row"):
        _strategy_nd_feature_half_y_reduce(
            valid[:, :, :-1], feature_half=0
        )
    with np.testing.assert_raises_regex(ValueError, "rounded BF16"):
        _strategy_nd_feature_half_y_reduce(
            valid.astype(jnp.float32), feature_half=0
        )
    with np.testing.assert_raises_regex(ValueError, "zero or one"):
        _strategy_nd_feature_half_y_reduce(valid, feature_half=2)
    with np.testing.assert_raises_regex(ValueError, "exact 0<->1 pairs"):
        _reduce_strategy_nd_feature_half_bf16_partials(
            valid,
            valid,
            axis_name="stage",
            pairs=((0, 1),),
        )


def test_pp16_feature2_dense_graph_has_only_half_output_and_one_exchange() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    stage_local_dense_feature2_fp8_mapped,
)

mesh = Mesh(np.asarray(jax.devices()), ("feature",))
def abstract(shape, dtype, spec):
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=NamedSharding(mesh, spec)
    )

arguments = (
    abstract((1, 6144), jnp.bfloat16, P()),
    abstract((2, 16, 6144, 768), jnp.uint8, P("feature", None, None, None)),
    abstract((2, 16, 48, 768), jnp.float32, P("feature", None, None, None)),
    abstract((2, 16, 384, 6144), jnp.uint8, P("feature", None, None, None)),
    abstract((2, 16, 3, 6144), jnp.float32, P("feature", None, None, None)),
)

def lower(full_width_rounded_then_slice):
    def mapped(normalized, merged_bits, merged_scale, down_bits, down_scale):
        value = stage_local_dense_feature2_fp8_mapped(
            normalized,
            merged_bits[0],
            merged_scale[0],
            down_bits[0],
            down_scale[0],
            axis_name="feature",
            pairs=((0, 1), (1, 0)),
            full_width_rounded_then_slice=full_width_rounded_then_slice,
        )
        return value[None, ...]

    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            P(),
            P("feature", None, None, None),
            P("feature", None, None, None),
            P("feature", None, None, None),
            P("feature", None, None, None),
        ),
        out_specs=P("feature", None, None),
        check_vma=False,
    )
    stablehlo = jax.jit(execute).lower(*arguments).as_text()
    public = next(
        line.strip() for line in stablehlo.splitlines()
        if "func.func public @main" in line
    )
    return {
        "all_gather_count": stablehlo.count("stablehlo.all_gather"),
        "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
        "collective_permute_count": stablehlo.count("stablehlo.collective_permute"),
        "convolution_count": stablehlo.count("stablehlo.convolution"),
        "full_hidden_stack": "tensor<16x1x6144xbf16>" in stablehlo,
        "half_output": "-> (tensor<2x1x3072xbf16>" in public,
        "payload_present": "tensor<4x1x3072xbf16>" in stablehlo,
    }

print(json.dumps({
    "default": lower(False),
    "full_width_then_slice": lower(True),
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
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    common = {
        "all_gather_count": 0,
        "all_reduce_count": 0,
        "collective_permute_count": 1,
        "full_hidden_stack": False,
        "half_output": True,
        "payload_present": True,
    }
    assert result == {
        "default": {**common, "convolution_count": 48},
        "full_width_then_slice": {**common, "convolution_count": 32},
    }


def test_pp16_final_layout_forced_two_device_graph_is_one_row_and_local() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.stage_local import (
    _reduce_strategy_nd_row0_bf16_partials,
    _virtual_dense_final_layout_convolution_down_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("stage",))
def abstract(shape, dtype, spec):
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=NamedSharding(mesh, spec)
    )

def mapped(normalized, merged_bits, merged_scale, down_bits, down_scale):
    partials = _virtual_dense_final_layout_convolution_down_partials(
        normalized,
        merged_bits[0],
        merged_scale[0],
        down_bits[0],
        down_scale[0],
        block_shape=(128, 128),
        compile_rows=1,
        virtual_shards=16,
    )
    return _reduce_strategy_nd_row0_bf16_partials(
        partials, axis_name="stage", groups=((0, 1),)
    )

execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(
        P(),
        P("stage", None, None, None),
        P("stage", None, None, None),
        P("stage", None, None, None),
        P("stage", None, None, None),
    ),
    out_specs=P(),
    check_vma=False,
)
arguments = (
    abstract((1, 6144), jnp.bfloat16, P()),
    abstract((2, 16, 6144, 768), jnp.uint8, P("stage", None, None, None)),
    abstract((2, 16, 48, 768), jnp.float32, P("stage", None, None, None)),
    abstract((2, 16, 384, 6144), jnp.uint8, P("stage", None, None, None)),
    abstract((2, 16, 3, 6144), jnp.float32, P("stage", None, None, None)),
)
stablehlo = jax.jit(execute).lower(*arguments).as_text()
print(json.dumps({
    "all_gather_count": stablehlo.count("stablehlo.all_gather"),
    "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
    "convolution_count": stablehlo.count("stablehlo.convolution"),
    "dead_rows": "tensor<32x6144xbf16>" in stablehlo,
    "payload_present": "tensor<4x1x6144xbf16>" in stablehlo,
}))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "all_gather_count": 0,
        "all_reduce_count": 1,
        "convolution_count": 32,
        "dead_rows": False,
        "payload_present": True,
    }
