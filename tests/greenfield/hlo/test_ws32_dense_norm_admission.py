"""Norm compiler-contract grammar mutations; not acquired TPU evidence."""

from dataclasses import replace
import json

import pytest

from scripts.greenfield import ws32_dense_norm_admission as admission
from scripts.greenfield import ws32_dense_frontier_admission as old
from scripts.greenfield import ws32_dense_norm_protocol as protocol


def fixture() -> str:
    text = [
        "HloModule norm, num_partitions=32",
        "%sum {\n%a = f32[] parameter(0)\n%b = f32[] parameter(1)\n"
        "ROOT %s = f32[] add(%a, %b)\n}",
        "ENTRY %main {",
    ]
    names, shapes = [], []
    for n in range(3):
        text.extend(
            [
                f"%h{n} = bf16[128,1536] constant(0)",
                f"%w{n} = u8[1536,1536] constant(0)",
                f"%s{n} = f32[16,128] constant(0)",
                f"%call{n} = f32[128,1536] custom-call(%h{n}, %w{n}, %s{n}), "
                f'custom_call_target="tpu_custom_call", metadata={{op_name="{admission.SUFFIX_SCOPE}"}}',
            ]
        )
        names.append(f"%call{n}")
        shapes.append("f32[128,1536]")
    for n, (groups, dims, scope) in enumerate(
        [
            (old.FEATURE, "2,128,1536", "feature_gate_up_reduce"),
            (old.EXPERT, "128,1536", "expert_down_reduce"),
        ]
    ):
        groups = "{" + ",".join("{" + ",".join(map(str, g)) + "}" for g in groups) + "}"
        text.extend(
            [
                f"%in{n} = f32[{dims}] constant(0)",
                f"%red{n} = bf16[{dims}] all-reduce(%in{n}), replica_groups={groups}, "
                f'use_global_device_ids=true, to_apply=%sum, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_prefill_dense/{scope}/psum"}}',
            ]
        )
        names.append(f"%red{n}")
        shapes.append(f"bf16[{dims}]")
    text.append(f'ROOT %out = ({", ".join(shapes)}) tuple({", ".join(names)})\n}}')
    return "\n".join(text)


def test_complete_synthetic_structure_and_report(monkeypatch):
    text = fixture()
    result = admission.inspect_suffix(text)
    assert result["kernels"]["kernel_count"] == 3
    assert result["collectives"]["static_instruction_count"] == 2
    # Only the raw pin is fixture-injected. The complete structural and memory
    # checks execute; no compiler result or model arithmetic is simulated here.
    from hashlib import sha256

    monkeypatch.setitem(protocol.RAW, "dense_suffix", (4, sha256(b"test").hexdigest()))
    memory = dict(old.MEMORY_CAPS)
    report = admission.inspect_program("dense_suffix", "test", text, memory)
    assert json.loads(json.dumps(report)) == report
    assert report["profile"] == protocol.PROFILE
    assert report["performance_claim"] is False
    assert report["numerical_promotion"] is False


@pytest.mark.parametrize(
    "case",
    [
        "scope",
        "extra",
        "dead",
        "alias",
        "dtype",
        "group",
        "reducer",
        "partitions",
        "host",
        "helper_shape",
        "helper_unknown",
        "scratch",
    ],
)
def test_structure_mutations(case):
    text = fixture()
    if case == "scope":
        text = text.replace(
            admission.SUFFIX_SCOPE, admission.SUFFIX_SCOPE.replace("body", "other"), 1
        )
    elif case in ("extra", "dead"):
        line = next(line for line in text.splitlines() if line.startswith("%call0 ="))
        text = text.replace(
            "ROOT %out", line.replace("%call0", "%extra") + "\nROOT %out"
        )
        if case == "extra":
            text = text.replace("tuple(%call0,", "tuple(%extra, %call0,")
            text = text.replace("ROOT %out = (", "ROOT %out = (f32[128,1536], ")
    elif case == "alias":
        text = text.replace(
            'custom_call_target="tpu_custom_call",',
            'custom_call_target="tpu_custom_call", output_to_operand_aliasing={{}: (0, {})},',
            1,
        )
    elif case == "dtype":
        text = text.replace("u8[1536,1536]", "bf16[1536,1536]")
    elif case == "group":
        text = text.replace("{0,1,2,3}", "{0,1,2,4}", 1)
    elif case == "reducer":
        text = text.replace("add(%a, %b)", "maximum(%a, %b)")
    elif case == "partitions":
        text = text.replace("num_partitions=32", "num_partitions=16")
    elif case == "host":
        text = text.replace("ROOT %out", "%host = s32[] infeed()\nROOT %out")
    else:
        if case == "scratch":
            size = 128
            helper = (
                '%helper = s32[128] custom-call(), custom_call_target="AllocateBuffer"'
            )
        else:
            size = 256 if case == "helper_shape" else 128
            target = (
                "AssumeGatherIndicesInBound"
                if case == "helper_shape"
                else "UnknownHelper"
            )
            helper = f'%idx = s32[{size}] constant(0)\n%helper = s32[{size}] custom-call(%idx), custom_call_target="{target}", metadata={{op_name="jit(body)/shard_map/gather"}}'
        text = text.replace("ROOT %out", helper + "\nROOT %out")
        text = text.replace("ROOT %out = (", f"ROOT %out = (s32[{size}], ")
        text = text.replace("tuple(%call0,", "tuple(%helper, %call0,")
    with pytest.raises(ValueError):
        admission.inspect_suffix(text)


def test_live_bounded_row_annotation():
    helper = (
        "%idx = s32[128] constant(0)\n"
        '%helper = s32[128] custom-call(%idx), custom_call_target="AssumeGatherIndicesInBound", '
        'metadata={op_name="jit(body)/shard_map/gather"}'
    )
    text = fixture().replace("ROOT %out", helper + "\nROOT %out")
    text = text.replace("ROOT %out = (", "ROOT %out = (s32[128], ")
    text = text.replace("tuple(%call0,", "tuple(%helper, %call0,")
    result = admission.inspect_suffix(text)
    assert result["helpers"]["passed"]
    assert result["helpers"]["bounded_annotation_counts"][0]["count"] == 1


def test_suffix_entry_and_resolver_type():
    module = old.parse_hlo_module(fixture())
    op = next(op for op in module.instructions if op.name == "%call0")
    assert admission.suffix_layer(op) == -1
    with pytest.raises(ValueError, match="ENTRY"):
        admission.suffix_layer(replace(op, computation="nested_body"))


@pytest.mark.parametrize("case", ["raw", "role", "memory", "alias_memory"])
def test_identity_memory_refusal(monkeypatch, case):
    from hashlib import sha256

    monkeypatch.setitem(protocol.RAW, "dense_suffix", (4, sha256(b"test").hexdigest()))
    memory = dict(old.MEMORY_CAPS)
    name, stable = "dense_suffix", "test"
    if case == "raw":
        stable = "wrong"
    elif case == "role":
        name = "dense01"
    elif case == "memory":
        memory["temp_size_in_bytes"] += 1
    else:
        memory["alias_size_in_bytes"] = 1
    with pytest.raises(ValueError):
        admission.inspect_program(name, stable, fixture(), memory)


def test_capture_and_wk_reuse_fixed_original_structure(monkeypatch):
    from hashlib import sha256

    calls = []
    monkeypatch.setattr(
        old, "inspect_structure", lambda name, text: calls.append((name, text)) or {}
    )
    for name in protocol.PROGRAMS[:3]:
        monkeypatch.setitem(protocol.RAW, name, (4, sha256(b"test").hexdigest()))
        report = admission.inspect_program(
            name, "test", "actual graph slot", dict(old.MEMORY_CAPS)
        )
        assert report["graph"] == name
    assert [name for name, _ in calls] == ["wk_decode", "wk_promote", "dense01"]
