"""Deep complete-model graphs must not depend on Python's recursion limit."""

import sys

import pytest

from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


def live(text):
    return _live_instruction_closure(parse_hlo_module(text).instructions)


def test_deep_local_operand_chain():
    depth = sys.getrecursionlimit() + 100
    operations = ["  %x0 = f32[] parameter(0)"]
    operations.extend(f"  %x{i} = f32[] copy(%x{i-1})" for i in range(1, depth))
    operations.append(f"  ROOT %out = f32[] copy(%x{depth-1})")
    result = live("HloModule deep\nENTRY main {\n" + "\n".join(operations) + "\n}")
    assert len(result) == depth + 1
    assert [op.index for op in result] == sorted(op.index for op in result)


@pytest.mark.parametrize("opcode,attribute", [("fusion", "calls"), ("call", "calls")])
def test_deep_callee_chain(opcode, attribute):
    depth = sys.getrecursionlimit() + 100
    parts = ["HloModule deep"]
    parts.append("%f0 {\n  ROOT %p0 = f32[] parameter(0)\n}")
    for i in range(1, depth):
        parts.append(
            f"%f{i} {{\n  %p{i} = f32[] parameter(0)\n"
            f"  ROOT %r{i} = f32[] {opcode}(%p{i}), {attribute}=%f{i-1}\n}}"
        )
    parts.append(
        "ENTRY main {\n  %input = f32[] parameter(0)\n"
        f"  ROOT %result = f32[] {opcode}(%input), {attribute}=%f{depth-1}\n}}"
    )
    result = live("\n".join(parts))
    assert len(result) == 2 * depth + 1


FUSION = """HloModule live
%identity {
  ROOT %used = f32[] parameter(0)
  %unused = f32[] parameter(1)
}
%nested {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %inner = f32[] fusion(%p0, %p1), kind=kLoop, calls=%identity
}
%sum {
  %s0 = f32[] parameter(0)
  %s1 = f32[] parameter(1)
  ROOT %add = f32[] add(%s0, %s1)
}
ENTRY main {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  %decoy = f32[] all-reduce(%x), replica_groups={{0,1}}, to_apply=%sum
  %left = f32[] fusion(%x, %decoy), kind=kLoop, calls=%nested
  %right = f32[] fusion(%y, %decoy), kind=kLoop, calls=%nested
  ROOT %out = f32[] add(%left, %right)
}
"""


def test_nested_unused_argument_collective_remains_dead_and_shared_calls_are_live():
    result = live(FUSION)
    assert {op.name for op in result} == {
        "%used",
        "%p0",
        "%inner",
        "%x",
        "%y",
        "%left",
        "%right",
        "%out",
    }


@pytest.mark.parametrize(
    "before,after,message",
    [
        ("calls=%identity", "calls=%nested", "recursive HLO computation"),
        ("calls=%identity", "calls=%missing", "has no root"),
        ("fusion(%p0, %p1)", "fusion()", "omits a live argument"),
        ("add(%left, %right)", "add(%left, %missing)", "undefined live HLO value"),
    ],
)
def test_malformed_live_graph_still_refuses(before, after, message):
    with pytest.raises(ValueError, match=message):
        live(FUSION.replace(before, after))


def test_control_flow_callees_materialized_iteratively():
    result = live(
        """HloModule control
%body {
  ROOT %b = f32[] parameter(0)
}
%cond {
  ROOT %c = pred[] constant(false)
}
%left {
  ROOT %l = f32[] parameter(0)
}
%right {
  ROOT %r = f32[] parameter(0)
}
ENTRY main {
  %x = f32[] parameter(0)
  %pred = pred[] constant(true)
  %loop = f32[] while(%x), condition=%cond, body=%body
  ROOT %choose = f32[] conditional(%pred, %loop, %x), branch_computations={%left,%right}
}
"""
    )
    assert {op.name for op in result} == {
        "%b",
        "%c",
        "%l",
        "%r",
        "%x",
        "%pred",
        "%loop",
        "%choose",
    }
