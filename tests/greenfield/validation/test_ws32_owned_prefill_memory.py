"""Owned-state arithmetic plus actual CPU compiled handles and host execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillResult, Ws32BatchedPrefillState
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState
from glm_tpu.greenfield.validation import ws32_prefill_memory as original
from scripts.greenfield import ws32_owned_prefill_memory as memory


def record(*, shared=True):
    analysis = dict(argument_size_in_bytes=100, output_size_in_bytes=82,
                    alias_size_in_bytes=80, temp_size_in_bytes=50,
                    generated_code_size_in_bytes=10)
    value = dict(schema_version=memory.SCHEMA, ownership_contract=memory.CONTRACT,
        donated_argument=2, donated_state_leaves=12,
        executable_roles={r: memory.ROLES[0] if shared else r for r in memory.ROLES},
        compiled_memory={r: dict(analysis) for r in memory.ROLES[:1 if shared else 2]},
        required_reserve_bytes=100,
        census=dict(schema_version=original.SCHEMA, includes_all_live_arrays=True,
            devices=[dict(device_id=0, buffers=[dict(bytes=80, identity_mode="physical_pointer",
                    groups=["active_state", "active_inputs", "__all_live_arrays__"]),
                dict(bytes=60, identity_mode="physical_pointer", groups=["__all_live_arrays__"])],
                accounted_resident_bytes=140,
                memory_stats=dict(bytes_in_use=150, peak_bytes_in_use=160, bytes_limit=400))]))
    value["budgets"] = memory.budgets(value)
    return value


def test_only_actual_active_alias_removed_all_live_reserve_and_code_retained():
    for shared, code in ((True, 10), (False, 20)):
        value = record(shared=shared)
        memory.validate_record(value)
        budget = value["budgets"][memory.ROLES[0]]["allocation_budget"]["devices"][0]
        assert budget["estimated_peak_bytes"] == 150 + code + 2 + 50
        assert budget["resident_baseline_bytes"] == 150
        assert value["compiled_memory"][memory.ROLES[0]]["output_size_in_bytes"] == 82
        with pytest.raises(ValueError, match="no-donation"):
            original.budget_resident_execution(value["census"], value["compiled_memory"],
                active_graph=memory.ROLES[0], resident_graphs=tuple(value["compiled_memory"]),
                required_reserve_bytes=100)


@pytest.mark.parametrize("mutation", ["missing_state", "state_weight_alias", "unknown_pointer",
                                      "too_much_alias", "no_alias", "prior_peak", "reserve", "forged_pass"])
def test_refusals_preserve_ownership_and_real_budget(mutation):
    value = record()
    state = value["census"]["devices"][0]["buffers"][0]
    if mutation == "missing_state":
        state["groups"].remove("active_state")
    elif mutation == "state_weight_alias":
        state["groups"].append("nonstate_inputs")
    elif mutation == "unknown_pointer":
        state["identity_mode"] = "distinct_object_upper_count"
    elif mutation == "too_much_alias":
        value["compiled_memory"][memory.ROLES[0]]["alias_size_in_bytes"] = 81
    elif mutation == "no_alias":
        value["compiled_memory"][memory.ROLES[0]]["alias_size_in_bytes"] = 0
    elif mutation == "prior_peak":
        value["census"]["devices"][0]["memory_stats"]["peak_bytes_in_use"] = 350
        value["budgets"] = memory.budgets(value)
    elif mutation == "reserve":
        value["required_reserve_bytes"] = 300
        value["budgets"] = memory.budgets(value)
    else:
        value["budgets"][memory.ROLES[0]]["allocation_budget"]["devices"][0]["estimated_peak_bytes"] = 1
    with pytest.raises(ValueError):
        memory.validate_record(value)


class Stats:
    """CPU allocator counters are fixture values, not measured TPU HBM."""
    def __init__(self, device):
        self.id, self.platform, self.process_index = device.id, device.platform, device.process_index

    def memory_stats(self):
        return dict(bytes_in_use=1 << 20, peak_bytes_in_use=1 << 20, bytes_limit=1 << 30)


def state(length=35):
    d = Ws32DecoderState(jnp.zeros(4), jnp.ones(4), jnp.zeros((1, 1), jnp.int32),
        jnp.zeros(1, jnp.int32), jnp.zeros((1, 1)), jnp.array([0], jnp.int32),
        jnp.array([[0]], jnp.int32), jnp.array([1], jnp.int32), jnp.array([True]))
    return Ws32BatchedPrefillState(d, jnp.full(4, 2.), jnp.int32(length), jnp.bool_(False))


def body(tokens, count, current, weights, wk, rope):
    end = current.decoder.position + count
    final = end[0] == current.prompt_length
    decoder = current.decoder._replace(kv_cache_local=current.decoder.kv_cache_local + weights[0],
        position=end, context_lengths=end + 1)
    return Ws32BatchedPrefillResult(current._replace(decoder=decoder, finished=final),
                                   jnp.where(final, 123, -1)[None])


def test_actual_compiled_object_identity_donation_and_state_weight_alias_refusal():
    inputs = (jnp.zeros(17, jnp.int32), jnp.int32(17), state(), jnp.ones(1), (jnp.ones(1),), jnp.ones(1))
    program = jax.jit(body, donate_argnums=(2,))
    first = program.lower(*inputs).compile()
    # Distinct Python compiled objects can wrap the same cached executable. We
    # deliberately overcount them, rather than infer sharing from equal HLO.
    second = program.lower(*inputs).compile()
    assert first is not second
    devices = [Stats(d) for d in jax.local_devices()]
    for other, count in ((first, 1), (second, 2)):
        value = memory.make_record(dict(zip(memory.ROLES, (first, other))), inputs,
            devices=devices, required_reserve_bytes=100)
        assert len(value["compiled_memory"]) == count
        memory.validate_record(value)
    undonated = jax.jit(body).lower(*inputs).compile()
    with pytest.raises(ValueError, match="donation"):
        memory.make_record(dict.fromkeys(memory.ROLES, undonated), inputs,
                           devices=devices, required_reserve_bytes=100)
    bad = (*inputs[:3], inputs[2].decoder.kv_cache_local, *inputs[4:])
    bad_compiled = program.lower(*bad).compile()
    with pytest.raises(ValueError, match="aliases nonstate"):
        memory.make_record(dict.fromkeys(memory.ROLES, bad_compiled), bad,
                           devices=devices, required_reserve_bytes=100)


def test_real_host_adapter_uses_owned_budget_then_consumed_state(monkeypatch):
    from scripts.greenfield import ws32_batched_prefill_runner as adapter
    from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config
    plan = adapter.BatchedPrefillPlan(35, 17, 8192)
    held = []
    def fresh(*args, **kwargs):
        value = state()
        held.append(value.decoder.kv_cache_local)
        return value
    monkeypatch.setattr(adapter, "make_ws32_batched_prefill_state", fresh)
    monkeypatch.setattr(adapter, "replicated", lambda mesh, value: jnp.asarray(value))
    capture = original.capture_resident_buffers
    monkeypatch.setattr(original, "capture_resident_buffers", lambda roots, *, devices:
        capture(roots, devices=[Stats(d) for d in devices]))
    weights, wk, rope = jnp.ones(1), (jnp.ones(1),), jnp.ones(1)
    placeholder = state()
    program = jax.jit(body, donate_argnums=(2,))
    compiled = {name: program.lower(jnp.zeros(rows, jnp.int32), jnp.int32(rows),
        placeholder, weights, wk, rope).compile() for name, rows in plan.graph_rows}
    values = []
    decoder, token, result = adapter.execute_graph_pair(None, config(), plan,
        np.arange(35, dtype=np.int32), compiled, weights, wk, rope,
        budget_seconds=60, required_memory_reserve_bytes=100,
        progress=values.append, fleet_all=bool, state_ownership_contract=memory.CONTRACT)
    assert held[0].is_deleted() and len(values) == 3
    assert decoder.position.tolist() == [35] and token.tolist() == [123]
    assert result["memory_admission"]["schema_version"] == memory.SCHEMA
    memory.validate_record(result["memory_admission"])
