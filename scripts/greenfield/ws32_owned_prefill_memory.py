"""Distinct consumed-state budget; historical no-donation admission stays strict.

Uses actual compiled alias bytes, actual argument donation flags, one all-live
census, and the original conservative allocator/code/scratch/reserve arithmetic.
Two roles share code accounting ONLY when they reference the same compiled Python
object at capture time. Equal graphs or equal memory reports are insufficient.
This is not HLO authorization or measured numerical-peak admission.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

import jax

from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillState
from glm_tpu.greenfield.validation import ws32_prefill_memory as original
from scripts.greenfield.ws32_prefill_owned_state import CONTRACT


SCHEMA = "ws32_owned_prefill_memory_v1"
ROLES = ("prefill_chunk", "prefill_tail")


def budgets(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute allocation estimates, retaining actual compiler reports intact."""
    if record.get("schema_version") != SCHEMA or record.get("ownership_contract") != CONTRACT:
        raise ValueError("owned prefill memory contract differs")
    roles = record["executable_roles"]
    analyses = record["compiled_memory"]
    if (set(roles) != set(ROLES) or set(roles.values()) != set(analyses)
            or roles[ROLES[0]] != ROLES[0]
            or roles[ROLES[1]] not in ROLES):
        raise ValueError("owned prefill resident roles differ")
    if (type(record.get("donated_argument")) is not int or record["donated_argument"] != 2
            or type(record.get("donated_state_leaves")) is not int or record["donated_state_leaves"] <= 0):
        raise ValueError("owned prefill state donation differs")
    census = record["census"]
    for device in census.get("devices", []):
        state_bytes = 0
        for entry in device["buffers"]:
            groups = set(entry.get("groups", []))
            if "active_state" in groups:
                if ("active_inputs" not in groups or "nonstate_inputs" in groups
                        or entry.get("identity_mode") != "physical_pointer"):
                    raise ValueError("donated state aliases nonstate or lacks physical ownership")
                state_bytes += original._integer(entry["bytes"], "state allocation", positive=True)
        if state_bytes == 0:
            raise ValueError("owned prefill census omits active state")
        for analysis in analyses.values():
            alias = original._integer(analysis.get("alias_size_in_bytes"), "state alias", positive=True)
            if alias > min(state_bytes, analysis["output_size_in_bytes"], analysis["argument_size_in_bytes"]):
                raise ValueError("compiled aliases exceed accounted state/arguments/output")
    result = {}
    for role in ROLES:
        graph = roles[role]
        effective = deepcopy(analyses)
        alias = effective[graph]["alias_size_in_bytes"]
        # Only the ACTIVE output allocation is reduced, never live baseline,
        # scratch, reserve, previous peak, or another executable's code.
        effective[graph]["output_size_in_bytes"] -= alias
        effective[graph]["alias_size_in_bytes"] = 0
        budget = original.budget_resident_execution(census, effective,
            active_graph=graph, resident_graphs=tuple(analyses),
            required_reserve_bytes=record["required_reserve_bytes"])
        result[role] = {"active_role": role, "compiled_state_alias_bytes": alias,
                        "allocation_budget": budget}
    return result


def make_record(compiled: Mapping[str, Any], inputs: tuple[Any, ...], *,
                devices: Sequence[Any], required_reserve_bytes: int) -> dict[str, Any]:
    """Snapshot actual donating executables and roots before the first dispatch.

    Caller must release repair/other model executables before this interface;
    it admits exactly the pair (or its single shared E0 object), not unlisted
    companion code. All other live ARRAY allocations are always included.
    """
    if set(compiled) != set(ROLES) or len(inputs) != 6 or type(inputs[2]) is not Ws32BatchedPrefillState:
        raise ValueError("owned prefill requires the actual six-argument state interface")
    count = len(jax.tree.leaves(inputs[2]))
    roles, resident, objects = {}, {}, {}
    for role in ROLES:
        executable = compiled[role]
        args, kwargs = executable.args_info
        if kwargs or len(args) != 6 or type(args[2]) is not Ws32BatchedPrefillState:
            raise ValueError("owned compiled argument structure differs")
        for index, arg in enumerate(args):
            leaves = jax.tree.leaves(arg)
            if (any(leaf.donated is not (index == 2) for leaf in leaves)
                    or (index == 2 and len(leaves) != count)):
                raise ValueError("compiled donation is not exclusively the complete state")
        for actual, abstract in zip(jax.tree.leaves(inputs[2]), jax.tree.leaves(executable.in_avals[0][2]), strict=True):
            if actual.shape != abstract.shape or actual.dtype != abstract.dtype:
                raise ValueError("compiled state shape/dtype differs from active state")
        canonical = objects.setdefault(id(executable), role)
        roles[role] = canonical
        if canonical not in resident:
            memory = executable.memory_analysis()
            resident[canonical] = {key: getattr(memory, key, None) for key in original.MEMORY_FIELDS}
    census = original.capture_resident_buffers(
        {"active_inputs": inputs, "active_state": inputs[2],
         "nonstate_inputs": (*inputs[:2], *inputs[3:])}, devices=devices)
    record = dict(schema_version=SCHEMA, ownership_contract=CONTRACT,
                  donated_argument=2, donated_state_leaves=count,
                  executable_roles=roles, compiled_memory=resident, census=census,
                  required_reserve_bytes=required_reserve_bytes)
    record["budgets"] = budgets(record)
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    """Sealer-side arithmetic replay, not trust in a reported fit boolean."""
    if set(record) != {"schema_version", "ownership_contract", "donated_argument",
                      "donated_state_leaves", "executable_roles", "compiled_memory",
                      "census", "required_reserve_bytes", "budgets"}:
        raise ValueError("owned prefill memory record fields differ")
    expected = budgets(record)
    if (json.dumps(expected, sort_keys=True, allow_nan=False)
            != json.dumps(record["budgets"], sort_keys=True, allow_nan=False)
            or not all(value["allocation_budget"]["estimate_fits"] for value in expected.values())):
        raise ValueError("owned prefill estimate fails or record drifted")
