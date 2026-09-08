"""Boolean implication regressions, including padded domains and array choices."""

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_hlo_boolean_factors import BooleanFactors
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


def setup(body, prefix="", *, live=False, nonempty=True):
    index = PrefillHloIndex(
        parse_hlo_module(
            "HloModule health\n" + prefix + "\nENTRY %main {\n" + body + "\n}"
        )
    )
    entry = index.computations["ENTRY"]
    engine = BooleanFactors(
        index,
        live_rows=2,
        live_mask=lambda value: (
            0 if live and value == engine.ref(entry["%live"]) else None
        ),
        nonempty_live=nonempty,
    )
    return engine, {name: engine.ref(op) for name, op in entry.items()}


def test_scalar_conjunction_and_disjunction_differ():
    e, v = setup(
        """%p = pred[] parameter(0)
%q = pred[] parameter(1)
%both = pred[] and(%p,%q)
ROOT %either = pred[] or(%p,%q)"""
    )
    both, either = e.factors(v["%both"]), e.factors(v["%either"])
    assert e.implies(both, v["%p"]) and e.implies(both, v["%q"])
    assert e.implies(both, v["%either"])
    assert not e.implies(either, v["%p"])
    assert not e.implies(either, v["%q"])
    assert not e.implies(frozenset(), v["%either"])


@pytest.mark.parametrize("choice", ["or(%x,%t)", "select(%selector,%x,%t)"])
def test_array_choice_cannot_intersect_coverage_only_factors(choice):
    e, v = setup(
        """%x = pred[2,2] parameter(0)
%selector = pred[2,2] parameter(1)
%t = pred[2,2] transpose(%x), dimensions={1,0}
ROOT %out = pred[2,2] """
        + choice
    )
    # x=[[T,F],[T,T]]: x OR transpose(x) is ALL true, x is not.
    assert not e.implies(e.factors(v["%out"]), v["%x"])
    assert e.implies(e.factors(v["%x"]), v["%out"])


@pytest.mark.parametrize("nonempty", [False, True])
def test_live_mask_never_promotes_all_padded_rows(nonempty):
    e, v = setup(
        """%live = pred[2] parameter(0)
%health = pred[2] parameter(1)
%scalar = pred[] parameter(2)
%broadcast = pred[2] broadcast(%scalar), dimensions={}
%notlive = pred[2] not(%live)
%masked = pred[2] or(%notlive,%health)
ROOT %masked_scalar = pred[2] or(%notlive,%broadcast)""",
        live=True,
        nonempty=nonempty,
    )
    known = e.factors(v["%masked"])
    assert e.implies(known, v["%health"], 0)
    assert not e.implies(known, v["%health"])
    assert e.implies(e.factors(v["%masked_scalar"]), v["%scalar"]) == nonempty
    assert e.implies(e.factors(v["%health"]), v["%health"], 0)


REDUCER = """%reduce_and {
%a = pred[] parameter(0)
%b = pred[] parameter(1)
ROOT %result = pred[] and(%a,%b)
}
"""
REDUCE_BODY = """%health = pred[2] parameter(0)
%init = pred[] constant(true)
ROOT %out = pred[] reduce(%health,%init), dimensions={0}, to_apply=%reduce_and"""


def test_and_reduction_equivalence():
    e, v = setup(REDUCE_BODY, REDUCER)
    assert e.implies(e.factors(v["%out"]), v["%health"])
    assert e.implies(e.factors(v["%health"]), v["%out"])


@pytest.mark.parametrize(
    "where,old,new",
    [
        ("reducer", "and(%a,%b)", "or(%a,%b)"),
        ("body", "constant(true)", "constant(false)"),
        ("body", "dimensions={0}", "dimensions={1}"),
    ],
)
def test_reduce_drift_refuses(where, old, new):
    body, reducer = REDUCE_BODY, REDUCER
    if where == "body":
        body = body.replace(old, new)
    else:
        reducer = reducer.replace(old, new)
    e, v = setup(body, reducer)
    with pytest.raises(ValueError):
        e.factors(v["%out"])


def test_fusion_call_binding_and_wrong_tuple_leaf():
    e, v = setup(
        """%a = pred[] parameter(0)
%b = pred[] parameter(1)
%c = (pred[],pred[]) fusion(%a,%b), calls=%pair
%d = (pred[],pred[]) fusion(%b,%a), calls=%pair
%first = pred[] get-tuple-element(%c), index=0
%second = pred[] get-tuple-element(%d), index=0
ROOT %out = pred[] and(%first,%second)""",
        """%pair {
%p = pred[] parameter(0)
%q = pred[] parameter(1)
ROOT %out = (pred[],pred[]) tuple(%q,%p)
}""",
    )
    assert e.implies(e.factors(v["%first"]), v["%b"])
    assert not e.implies(e.factors(v["%first"]), v["%a"])
    assert e.implies(e.factors(v["%second"]), v["%a"])
    assert e.implies(e.factors(v["%out"]), v["%a"])


def test_unknown_operation_cannot_prove_its_inputs():
    e, v = setup(
        """%p = pred[] parameter(0)
ROOT %out = pred[] custom-call(%p), custom_call_target="unknown"
"""
    )
    assert not e.implies(e.factors(v["%out"]), v["%p"])
    assert not e.implies(e.factors(v["%p"]), v["%out"])


def test_explicit_stack_handles_deep_boolean_graph():
    lines = ["%p = pred[] parameter(0)", "%q = pred[] parameter(1)"]
    last = "%p"
    for j in range(1500):
        name = f"%n{j}"
        lines.append(f"{'ROOT ' if j == 1499 else ''}{name} = pred[] and({last},%q)")
        last = name
    e, v = setup("\n".join(lines))
    assert e.implies(e.factors(v[last]), v["%p"])
    assert e.implies(e.factors(v["%p"]) | e.factors(v["%q"]), v[last])


def test_wrong_broadcast_dimensions_refuses():
    e, v = setup(
        """%p = pred[2] parameter(0)
ROOT %out = pred[2,3] broadcast(%p), dimensions={1}"""
    )
    with pytest.raises(ValueError, match="broadcast"):
        e.factors(v["%out"])


def test_unacquired_array_bitcast_remains_opaque_even_for_all_domain():
    e, v = setup(
        """%p = pred[2,2] parameter(0)
ROOT %out = pred[4] bitcast(%p)"""
    )
    assert not e.implies(e.factors(v["%out"]), v["%p"])
    assert not e.implies(e.factors(v["%p"]), v["%out"])
