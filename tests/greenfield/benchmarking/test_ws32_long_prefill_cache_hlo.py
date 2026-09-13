"""Bounded synthetic HLO lineage mutations; no large arrays are allocated."""

from collections import Counter

import pytest

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import _check_kernel_schedule
from glm_tpu.greenfield.benchmarking.ws32_long_prefill_cache_hlo import prove_large_cache_storage


def fixture(capacity=262656):
    shape = f"bf16[78,{capacity//512},64,640]"
    args = ", ".join(["s32[128]", "s32[]", shape] + ["s32[]"] * 11)
    return f'''HloModule cache, num_partitions=32
%write {{
 %base = {shape} parameter(0)
 %update = bf16[1,1,1,640] parameter(1)
 %zero = s32[] constant(0)
 ROOT %changed = {shape} dynamic-update-slice(%base, %update, %zero, %zero, %zero, %zero)
}}
%no {{
 %p = ({shape}, s32[]) parameter(0)
 %kv = {shape} get-tuple-element(%p), index=0
 %n = s32[] get-tuple-element(%p), index=1
 ROOT %out = ({shape}, s32[]) tuple(%kv, %n)
}}
%yes {{
 %p = ({shape}, s32[]) parameter(0)
 %kv = {shape} get-tuple-element(%p), index=0
 %n = s32[] get-tuple-element(%p), index=1
 %u = bf16[1,1,1,640] constant(0)
 %f = {shape} fusion(%kv, %u), kind=kLoop, calls=%write
 ROOT %out = ({shape}, s32[]) tuple(%f, %n)
}}
ENTRY %main {{
 %args = ({args}) parameter(0)
 %cache = {shape} get-tuple-element(%args), index=2
 %copy = {shape} copy(%cache)
 %z = s32[] constant(0)
 %input = ({shape}, s32[]) tuple(%copy, %z)
 %pred = pred[] constant(true)
 ROOT %out = ({shape}, s32[]) conditional(%pred, %input, %input), branch_computations={{%no, %yes}}
}}
'''


def check(text, capacity=262656):
    index = PrefillHloIndex(parse_hlo_module(text))
    report, approved = prove_large_cache_storage(index, context_capacity=capacity)
    assert report["passed"] and report["instruction_count"] == len(approved)
    return index, approved


def test_benchmark_capacity_requires_explicit_profile_without_widening_old_guard():
    index = PrefillHloIndex(parse_hlo_module(fixture(166912)))
    with pytest.raises(ValueError, match="registered capacity"):
        prove_large_cache_storage(index, context_capacity=166912)
    report, approved = prove_large_cache_storage(
        index, context_capacity=166912, native_benchmark=True)
    assert report['passed'] and approved
    assert {'dtype': 'bf16', 'dimensions': [78, 326, 64, 640]} in report['allowed_shapes']
    with pytest.raises(ValueError, match="registered capacity"):
        prove_large_cache_storage(index, context_capacity=167424, native_benchmark=True)


@pytest.mark.parametrize("capacity", [131072, 262656])
def test_real_storage_forms_and_original_short_refusal(capacity):
    index, approved = check(fixture(capacity), capacity)
    old = _check_kernel_schedule(index, live_instructions=index.module.instructions, expected=Counter())
    assert old["passed"] is False and "floating weight expansion" in old["error"]
    new = _check_kernel_schedule(index, live_instructions=index.module.instructions,
                                expected=Counter(), large_float_guard=lambda op, shape: op.index in approved)
    assert new["passed"]


@pytest.mark.parametrize("case", ["entry_slot", "entry_parameter", "extraction", "nested", "floating_weight",
                                 "cache_shaped_conversion", "update_base", "full_update", "branch_zero", "unbound"])
def test_storage_mutations_refuse(case):
    text = fixture()
    shape = "bf16[78,513,64,640]"
    if case == "entry_slot":
        text = text.replace(f"s32[128], s32[], {shape}", f"{shape}, s32[], s32[128]")
    elif case == "entry_parameter":
        head, tail = text.split("ENTRY %main", 1)
        text = head + "ENTRY %main" + tail.replace("parameter(0)", "parameter(1)")
    elif case == "extraction":
        text = text.replace("get-tuple-element(%args), index=2", "get-tuple-element(%args), index=0")
    elif case == "nested":
        text = text.replace(f"({shape}, s32[])", f"(({shape}), s32[])")
    elif case == "floating_weight":
        text = text.replace(" %z =", " %weight = bf16[32,2048,1536] constant(0)\n %z =")
    elif case == "cache_shaped_conversion":
        text = text.replace(f"%copy = {shape} copy(%cache)", f"%copy = {shape} convert(%cache)")
    elif case == "update_base":
        text = text.replace("dynamic-update-slice(%base, %update", "dynamic-update-slice(%update, %update")
    elif case == "full_update":
        text = text.replace("dynamic-update-slice(%base, %update", "dynamic-update-slice(%base, %base")
    elif case == "branch_zero":
        text = text.replace(f"%kv = {shape} get-tuple-element(%p), index=0", f"%kv = {shape} broadcast(%n)")
    else:
        text = text.replace("calls=%write", "calls=%yes")
    with pytest.raises((ValueError, KeyError, IndexError)):
        check(text)


@pytest.mark.parametrize("bad", [None, True, 8192, 131072.0, 262144])
def test_no_arbitrary_long_capacity(bad):
    with pytest.raises(ValueError, match="registered capacity"):
        prove_large_cache_storage(None, context_capacity=bad)
