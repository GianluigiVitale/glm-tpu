from __future__ import annotations

from functools import partial
import re

import jax
import jax.numpy as jnp
import pytest

from glm_tpu.greenfield.benchmarking.legacy_prefill_chunk_probe import (
    legacy_geometry_chunk_consumer_gather_from_normalized,
)
from glm_tpu.greenfield.validation.chunk0_real_layer_consumer_hlo import (
    require_real_layer_consumer_hlo,
)
from glm_tpu.greenfield.validation.original_db518_normalized_boundary_hlo import (
    _boundary_entry,
    _entry_sources,
)


def _consumer_avals():
    shape = jax.ShapeDtypeStruct
    weights = {
        "q_a_norm": shape((2048,), jnp.bfloat16),
        "kv_a_norm": shape((512,), jnp.bfloat16),
        "post_norm": shape((6144,), jnp.bfloat16),
        "input_norm1": shape((6144,), jnp.bfloat16),
        "k_norm1_weight": shape((128,), jnp.bfloat16),
        "k_norm1_bias": shape((128,), jnp.bfloat16),
        "qkv_bits": shape((32, 6144, 82), jnp.uint8),
        "qkv_scale": shape((32, 48, 82), jnp.float32),
        "qb_bits": shape((32, 2048, 512), jnp.uint8),
        "qb_scale": shape((32, 16, 4), jnp.float32),
        "w_uk_t": shape((32, 2, 192, 512), jnp.bfloat16),
        "w_uv": shape((32, 2, 512, 256), jnp.bfloat16),
        "o_bits": shape((32, 512, 6144), jnp.uint8),
        "o_scale": shape((32, 4, 48), jnp.float32),
        "gu_bits": shape((32, 6144, 768), jnp.uint8),
        "gu_scale": shape((32, 48, 6), jnp.float32),
        "down_bits": shape((32, 384, 6144), jnp.uint8),
        "down_scale": shape((32, 3, 48), jnp.float32),
        "wk1": shape((128, 6144), jnp.float32),
        "rope_table": shape((256, 64), jnp.bfloat16),
    }
    return (
        shape((37, 6144), jnp.bfloat16),
        shape((2048,), jnp.int32),
        shape((2048, 6144), jnp.bfloat16),
        shape((2048,), jnp.int32),
        weights,
    )


@pytest.fixture(scope="module")
def consumer_hlo():
    function = partial(
        legacy_geometry_chunk_consumer_gather_from_normalized,
        softmax_scale=0.0625,
    )
    lowered = jax.jit(function).lower(*_consumer_avals())
    return lowered.compile().as_text(), lowered.as_text()


def _replace_root_operand(optimized: str, source: int, target: int) -> str:
    lines = optimized.splitlines(keepends=True)
    candidates = [
        index
        for index, line in enumerate(lines)
        if line.startswith("  ROOT %tuple.")
        and " tuple(" in line
        and len(re.findall(r"%[A-Za-z0-9_.-]+", line.split("tuple(", 1)[1])) == 11
    ]
    assert len(candidates) == 1
    root_index = candidates[0]
    line = lines[root_index]
    start = line.index("tuple(") + len("tuple(")
    end = line.index(")", start)
    operands = [
        re.sub(r"^/\*index=[0-9]+\*/", "", item) for item in line[start:end].split(", ")
    ]
    assert len(operands) == 11
    operands[target] = operands[source]
    lines[root_index] = line[:start] + ", ".join(operands) + line[end:]
    return "".join(lines)


def _entry_root(optimized: str) -> tuple[list[str], int, list[str]]:
    lines = optimized.splitlines(keepends=True)
    candidates = [
        index
        for index, line in enumerate(lines)
        if line.startswith("  ROOT %tuple.")
        and " tuple(" in line
        and len(re.findall(r"%[A-Za-z0-9_.-]+", line.split("tuple(", 1)[1])) == 11
    ]
    assert len(candidates) == 1
    root_index = candidates[0]
    line = lines[root_index]
    start = line.index("tuple(") + len("tuple(")
    end = line.index(")", start)
    operands = [
        re.sub(r"^/\*index=[0-9]+\*/", "", item) for item in line[start:end].split(", ")
    ]
    assert len(operands) == 11
    return lines, root_index, operands


def _insert_key_wrapper(
    optimized: str,
    *,
    name: str,
    extra_operands: tuple[str, ...],
    include_live_key: bool = True,
) -> str:
    lines, root_index, operands = _entry_root(optimized)
    wrapped = (operands[5],) if include_live_key else ()
    node = (
        f"  %{name} = bf16[2048,128] custom-call("
        + ", ".join((*wrapped, *extra_operands))
        + f'), custom_call_target="{name}"\n'
    )
    lines.insert(root_index, node)
    mutated = "".join(lines)
    return _replace_root_operand_by_name(mutated, target=5, name=f"%{name}")


def _replace_root_operand_by_name(optimized: str, *, target: int, name: str) -> str:
    lines, root_index, operands = _entry_root(optimized)
    line = lines[root_index]
    start = line.index("tuple(") + len("tuple(")
    end = line.index(")", start)
    operands[target] = name
    lines[root_index] = line[:start] + ", ".join(operands) + line[end:]
    return "".join(lines)


def _entry_parameters(optimized: str) -> tuple[str, ...]:
    lines, root_index, _ = _entry_root(optimized)
    entry_start = max(
        index for index in range(root_index) if lines[index].startswith("ENTRY ")
    )
    parameters = []
    for line in lines[entry_start + 1 : root_index]:
        match = re.match(
            r"  (?P<name>%[A-Za-z0-9_.-]+) = .* parameter\((?P<number>[0-9]+)\)",
            line,
        )
        if match is not None:
            parameters.append((int(match.group("number")), match.group("name")))
    parameters.sort()
    assert [number for number, _ in parameters] == list(range(24))
    return tuple(name for _, name in parameters)


def _replace_qkv_activation_with_raw_embedding(optimized: str) -> str:
    lines = optimized.splitlines(keepends=True)
    raw = next(
        re.match(r"  (?P<name>%[A-Za-z0-9_.-]+) =", line).group("name")
        for line in lines
        if "bitcast_gather_fusion = f32[2048,1,6144]" in line and " fusion(" in line
    )
    candidates = [
        index
        for index, line in enumerate(lines)
        if " = (s32[], bf16[32,2048,82]" in line
        and " tuple(" in line
        and "u8[32,6144,82]" in line
        and "f32[32,48,82]" in line
        and "%weights__qkv_bits__.1" in line
        and "%weights__qkv_scale__.1" in line
    ]
    assert len(candidates) == 1
    index = candidates[0]
    line = lines[index]
    start = line.index("tuple(") + len("tuple(")
    end = line.index(")", start)
    operands = [
        re.sub(r"^/\*index=[0-9]+\*/", "", item) for item in line[start:end].split(", ")
    ]
    assert len(operands) == 6
    operands[4] = raw
    lines[index] = line[:start] + ", ".join(operands) + line[end:]
    return "".join(lines)


def _assert_source_only_contract_would_accept(optimized: str) -> None:
    computations, entry_name, root, params, _ = _boundary_entry(optimized)
    sources = tuple(
        frozenset(_entry_sources(computations, entry_name, operand, params))
        for operand in root["operands"]
    )
    assert sources[5] == frozenset(params.values())
    assert sources[7] == frozenset({params[2]})


def test_real_layer_consumer_hlo_accepts_exact_cpu_graph(consumer_hlo):
    optimized, stable = consumer_hlo
    report = require_real_layer_consumer_hlo(optimized, stable)
    assert report["passed"]
    assert report["optimized_keys1_sources_all_parameters"]
    assert report["optimized_qkv_projection"]["projection_kind"] == "body_dot"
    assert report["optimized_qkv_projection"][
        "selected_result_dominates_normalized_qkv_sources"
    ]
    assert report["stable_hidden_rms_reduce_count"] == 2
    assert report["stable_rsqrt_count"] == 4


def test_real_layer_consumer_hlo_rejects_keys1_boundary_bypass(consumer_hlo):
    optimized, stable = consumer_hlo
    bypassed = _replace_root_operand(optimized, source=7, target=5)
    with pytest.raises(RuntimeError):
        require_real_layer_consumer_hlo(bypassed, stable)


def test_real_layer_consumer_hlo_rejects_dead_qkv_and_all_input_root(consumer_hlo):
    optimized, stable = consumer_hlo
    alternate = _insert_key_wrapper(
        optimized,
        name="alternate_all_input_key",
        extra_operands=_entry_parameters(optimized),
        include_live_key=False,
    )
    _assert_source_only_contract_would_accept(alternate)
    with pytest.raises(RuntimeError, match="output-live QKV projection"):
        require_real_layer_consumer_hlo(alternate, stable)


def test_real_layer_consumer_hlo_rejects_normalized_decoy_around_qkv(consumer_hlo):
    optimized, stable = consumer_hlo
    normalized = _entry_parameters(optimized)[2]
    decoy = _insert_key_wrapper(
        optimized,
        name="normalized_decoy_key",
        extra_operands=(normalized,),
    )
    _assert_source_only_contract_would_accept(decoy)
    with pytest.raises(RuntimeError, match="path is bypassable"):
        require_real_layer_consumer_hlo(decoy, stable)


def test_real_layer_consumer_hlo_rejects_raw_qkv_with_normalized_decoy(consumer_hlo):
    optimized, stable = consumer_hlo
    normalized = _entry_parameters(optimized)[2]
    raw_qkv = _replace_qkv_activation_with_raw_embedding(optimized)
    decoy = _insert_key_wrapper(
        raw_qkv,
        name="normalized_decoy_for_raw_qkv",
        extra_operands=(normalized,),
    )
    _assert_source_only_contract_would_accept(decoy)
    with pytest.raises(RuntimeError, match="output-live QKV projection"):
        require_real_layer_consumer_hlo(decoy, stable)


def test_real_layer_consumer_hlo_rejects_input_rms_reintroduction(consumer_hlo):
    optimized, stable = consumer_hlo
    mutated = stable.replace("stablehlo.rsqrt", "stablehlo.rsqrt /* input_norm0 */", 1)
    assert mutated != stable
    with pytest.raises(RuntimeError):
        require_real_layer_consumer_hlo(optimized, mutated)


def test_real_layer_consumer_hlo_rejects_communication(consumer_hlo):
    optimized, stable = consumer_hlo
    with pytest.raises(RuntimeError, match="communication/callback"):
        require_real_layer_consumer_hlo(
            optimized + "\n%bad = all-reduce(%normalized0.1)\n", stable
        )
