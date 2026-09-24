"""Selected-leaf call binding and stack-safe dependency, never implication."""

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_hlo_leaf_dependencies import LeafDependencies
from glm_tpu.optimized.hlo_contract import parse_hlo_module


TEXT = """HloModule dependencies
%mix {
%a = f32[] parameter(0)
%b = f32[] parameter(1)
ROOT %out = (f32[],f32[]) tuple(%b,%a)
}
ENTRY %main {
%a = f32[] constant(10)
%b = f32[] constant(20)
%c = (f32[],f32[]) fusion(%a,%b), calls=%mix
%d = (f32[],f32[]) fusion(%b,%a), calls=%mix
%first = f32[] get-tuple-element(%c), index=0
%second = f32[] get-tuple-element(%d), index=0
ROOT %out = f32[] add(%first,%second)
}
"""


def setup(text=TEXT):
    index = PrefillHloIndex(parse_hlo_module(text))
    entry = index.computations["ENTRY"]
    a, b = entry["%a"], entry["%b"]
    return index, entry, LeafDependencies(index, {a.index, b.index})


def test_shared_fusion_summaries_bind_each_caller_and_selected_leaf():
    _, entry, deps = setup()
    assert deps.dependencies(entry["%first"]) == {("terminal", entry["%b"].index, ())}
    assert deps.dependencies(entry["%second"]) == {("terminal", entry["%a"].index, ())}
    assert deps.dependencies(entry["%out"]) == {
        ("terminal", entry[n].index, ()) for n in ("%a", "%b")
    }
    size = len(deps.memo)
    deps.dependencies(entry["%out"])
    assert len(deps.memo) == size


def test_wrong_leaf_changes_provenance():
    _, entry, deps = setup(TEXT.replace("%c), index=0", "%c), index=1"))
    assert deps.dependencies(entry["%first"]) == {("terminal", entry["%a"].index, ())}


def test_selected_nested_tuple_parameter():
    text = """HloModule nested
%select {
%p = (f32[],(f32[],f32[])) parameter(0)
%nested = (f32[],f32[]) get-tuple-element(%p), index=1
ROOT %selected = f32[] get-tuple-element(%nested), index=0
}
ENTRY %main {
%p = (f32[],(f32[],f32[])) parameter(0)
ROOT %out = f32[] fusion(%p), calls=%select
}
"""
    index = PrefillHloIndex(parse_hlo_module(text))
    deps = LeafDependencies(index, set())
    assert deps.dependencies(index.roots["ENTRY"]) == {("parameter", 0, (1, 0))}


def test_ancestry_does_not_prove_nonzero_contribution():
    _, entry, deps = setup(
        TEXT.replace(
            "ROOT %out = f32[] add(%first,%second)",
            "%zero = f32[] constant(0)\nROOT %out = f32[] multiply(%first,%zero)",
        )
    )
    assert deps.dependencies(entry["%out"]) == {("terminal", entry["%b"].index, ())}


def test_deep_chain_does_not_use_python_recursion():
    lines = ["HloModule deep", "ENTRY %main {", "%p = f32[] parameter(0)"]
    last = "%p"
    for i in range(3000):
        name = f"%v{i}"
        lines.append(f'{"ROOT " if i==2999 else ""}{name} = f32[] negate({last})')
        last = name
    index = PrefillHloIndex(parse_hlo_module("\n".join(lines) + "\n}"))
    assert LeafDependencies(index, set()).dependencies(index.roots["ENTRY"]) == {
        ("parameter", 0, ())
    }


@pytest.mark.parametrize(
    "replacement",
    [
        "ROOT %out = f32[] while(%first), body=%mix, condition=%mix",
        'ROOT %out = f32[] custom-call(%first), custom_call_target="unknown"',
        "ROOT %out = f32[] negate(%out)",
    ],
)
def test_unsupported_or_cyclic_dependency_refuses(replacement):
    _, entry, deps = setup(
        TEXT.replace("ROOT %out = f32[] add(%first,%second)", replacement)
    )
    with pytest.raises(ValueError):
        deps.dependencies(entry["%out"])
