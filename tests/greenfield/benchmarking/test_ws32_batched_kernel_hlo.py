"""Pallas interface mutation checks and integrated captured-profile replay."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import (
    _expected,
    check_batched_kernels,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import (
    UNREGISTERED,
    inspect_ws32_batched_prefill_hlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


def fixture(rows=17):
    def shape(s):
        return s[0] + "[" + ",".join(map(str, s[1])) + "]"

    lines = ["HloModule kernels, num_partitions=32", "ENTRY %main {"]
    n = 0
    for (layer, family, name, ins, outs, alias), count in _expected(rows).items():
        for _ in range(count):
            args = [f"%p{n}_{i}" for i in range(len(ins))]
            lines += [f"{arg} = {shape(s)} constant(0)" for arg, s in zip(args, ins)]
            extra = (
                f", output_to_operand_aliasing={{{{}}: ({alias}, {{}})}}"
                if alias >= 0
                else ""
            )
            lines.append(
                f'%call{n} = {shape(outs[0])} custom-call({", ".join(args)}), custom_call_target="tpu_custom_call"{extra}, metadata={{op_name="greenfield_ws32_batched_prefill/layer_{layer}/{name}/pallas_call"}}'
            )
            n += 1
    lines.append("ROOT %out = s32[] constant(0)\n}")
    return "\n".join(lines)


def check(module, rows=17, live=None):
    return check_batched_kernels(
        PrefillHloIndex(module),
        block_rows=rows,
        live_instructions=module.instructions if live is None else live,
    )


@pytest.fixture(scope="module")
def synthetic():
    return parse_hlo_module(fixture())


def mutate(module, old, **changes):
    return replace(
        module,
        instructions=tuple(
            replace(p, **changes) if p is old else p for p in module.instructions
        ),
    )


@pytest.mark.parametrize("rows", [11, 17])
def test_schedule_counts(rows):
    result = check(parse_hlo_module(fixture(rows)), rows)
    assert result["passed"], result
    assert result["kernel_count"] == 1047
    assert result["family_counts"] == dict(
        raw=588, grouped=225, structured=156, sparse=78
    )
    assert "OPAQUE_KERNEL_ARITHMETIC" in result["not_proven"]


@pytest.mark.parametrize(
    "case",
    [
        "weight_dtype",
        "scale_shape",
        "output_dtype",
        "missing",
        "extra",
        "dead",
        "kernel_name",
        "layer",
        "side_effect",
        "unexpected_alias",
        "arity",
    ],
)
def test_raw_refusal(synthetic, case):
    op = next(p for p in synthetic.instructions if p.raw_opcode == "custom-call")
    if case == "weight_dtype":
        ins = list(op.operand_shapes)
        ins[1] = replace(ins[1], dtype="bf16")
        m = mutate(synthetic, op, operand_shapes=tuple(ins))
    elif case == "scale_shape":
        ins = list(op.operand_shapes)
        ins[2] = replace(ins[2], dimensions=(128, 16))
        m = mutate(synthetic, op, operand_shapes=tuple(ins))
    elif case == "output_dtype":
        m = mutate(
            synthetic, op, result_shapes=(replace(op.result_shapes[0], dtype="bf16"),)
        )
    elif case == "missing":
        m = replace(
            synthetic,
            instructions=tuple(p for p in synthetic.instructions if p is not op),
        )
    elif case == "extra":
        m = replace(
            synthetic,
            instructions=synthetic.instructions + (replace(op, name="%extra"),),
        )
    elif case == "dead":
        assert not check(
            synthetic, live=tuple(p for p in synthetic.instructions if p is not op)
        )["passed"]
        return
    elif case == "kernel_name":
        m = mutate(synthetic, op, op_name=op.op_name.replace("m24_", "m16_"))
    elif case == "layer":
        m = mutate(synthetic, op, op_name=op.op_name.replace("/layer_0/", "/layer_3/"))
    elif case == "side_effect":
        m = mutate(
            synthetic,
            op,
            raw_line=op.raw_line.replace(
                ", metadata=", ", custom_call_has_side_effect=true, metadata="
            ),
        )
    elif case == "unexpected_alias":
        m = mutate(
            synthetic,
            op,
            raw_line=op.raw_line.replace(
                ", metadata=", ", output_to_operand_aliasing={{}: (0, {})}, metadata="
            ),
        )
    else:
        m = mutate(synthetic, op, operand_names=op.operand_names[:-1])
    assert not check(m)["passed"], case


@pytest.mark.parametrize("change", ["missing", "wrong", "duplicate"])
def test_grouped_alias_exact(synthetic, change):
    op = next(
        p
        for p in synthetic.instructions
        if p.op_name and "/greenfield_prefill_grouped_raw_fp8/" in p.op_name
    )
    alias = "output_to_operand_aliasing={{}: (8, {})}"
    text = op.raw_line.replace(
        alias,
        (
            ""
            if change == "missing"
            else (
                alias.replace("(8,", "(7,")
                if change == "wrong"
                else alias + ", " + alias
            )
        ),
    )
    assert not check(mutate(synthetic, op, raw_line=text))["passed"]


def test_full_expert_float_expansion_refuses(synthetic):
    op = synthetic.instructions[0]
    expansion = replace(
        op,
        name="%full_expansion",
        result_shapes=(
            replace(op.result_shapes[0], dtype="bf16", dimensions=(32, 2048, 1536)),
        ),
    )
    assert not check(
        replace(synthetic, instructions=synthetic.instructions + (expansion,))
    )["passed"]


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unregistered_rows(synthetic, rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check(synthetic, rows)


@pytest.mark.parametrize("graph,rows", [("prefill_chunk", 17), ("prefill_tail", 11)])
def test_original_integrated_profile(graph, rows):
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (
            root
            / "docs/artifacts/prefill-batched-seven-graph-acquisition-20260908.json"
        ).read_text()
    )
    folder = Path("/home/gianl/glm-run") / receipt["tag"] / "hlo"
    paths = [
        folder / (graph + "." + form)
        for form in ("stablehlo.mlir", "optimized_hlo.txt")
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("original HLO unavailable locally")
    raw = [p.read_bytes() for p in paths]
    pins = receipt["graphs"][graph]
    assert sha256(raw[0]).hexdigest() == pins["stablehlo_sha256"]
    assert sha256(raw[1]).hexdigest() == pins["optimized_hlo_sha256"]
    result = inspect_ws32_batched_prefill_hlo(
        raw[0].decode(),
        raw[1].decode(),
        block_rows=rows,
        expected_stablehlo_sha256=pins["stablehlo_sha256"],
        expected_optimized_hlo_sha256=pins["optimized_hlo_sha256"],
    )
    assert result["pallas_interface_proof"]["passed"], result["pallas_interface_proof"]
    assert result["operand_health_proof"]["passed"], result["operand_health_proof"]
    assert len(result["operand_health_proof"]["layers"]) == 78
    assert len(result["operand_health_proof"]["grouped_validity"]) == 225
    assert len(result["operand_health_proof"]["routers"]) == 75
    assert result["violations"] == [UNREGISTERED], result["violations"]
    assert not result["passed"] and not result["profile_registered"]
    from glm_tpu.greenfield.validation.ws32_prefill_admission import (
        SHORT_PROFILE,
        authorize_short_graph,
        short_graph_identity,
    )

    identity = short_graph_identity(
        raw[0].decode(),
        raw[1].decode(),
        graph=graph,
        profile=SHORT_PROFILE,
        repo=root,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
    )
    admitted = authorize_short_graph(
        {**result, "source_location_identity": identity},
        profile=SHORT_PROFILE,
        repo=root,
    )
    assert admitted["passed"] and admitted["profile_registered"]
    assert not admitted["runtime_memory_admitted"] and not admitted["numerical_claim"]
