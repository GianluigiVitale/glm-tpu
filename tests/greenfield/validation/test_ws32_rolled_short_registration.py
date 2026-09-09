"""Distinct combined-profile identity; registration never bypasses HLO admission."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from scripts.greenfield import ws32_batched_launch as launch

ROOT = Path(__file__).resolve().parents[3]
PROFILE = a.ROLLED_SHORT_PROFILE


def test_registered_recipe_source_targets_and_five_companions():
    a.require_acquired_model_source(ROOT, profile=PROFILE)
    record = a.rolled_registration(ROOT)
    plan = a.short_plan(PROFILE)
    assert plan.split == (15, 114) and plan.mlp_window
    assert plan.graph_rows == (("prefill_chunk", 128), ("prefill_tail", 114))
    assert record["program_options"] == a.short_program_options(PROFILE)
    identity = a.short_numerical_identity(profile=PROFILE)
    assert identity["batched_prefill_plan"] == plan.identity()
    assert identity["batched_prefill_program_options"] == record["program_options"]
    old = a.short_acquisition(ROOT)["graphs"]
    new = a.short_acquisition(ROOT, profile=PROFILE)["graphs"]
    for graph, pins in new.items():
        if graph.startswith("prefill_"):
            assert pins["optimized_hlo_sha256"] == a.FRESH_OPTIMIZED_MARKER
            assert (
                pins["stablehlo_sha256"] == record["graphs"][graph]["stablehlo_sha256"]
            )
        else:
            assert pins == old[graph]
    env = launch.numerical_environment(profile=PROFILE)
    assert env["GLM_GREENFIELD_WS32_PREFILL_CHUNK"] == "128"
    launch.validate_environment(env)


@pytest.mark.parametrize(
    "field,value",
    [
        ("PREFILL_CHUNK", "17"),
        ("PREFILL_CHUNK", "114"),
        ("PREFILL_CHUNK", "129"),
        ("CONTEXT_CAPACITY", "131072"),
        ("SHORT_DECODER_CONTEXT", "8k"),
        ("HOST_MAIN_ROPE_TABLE", "0"),
        ("EXACT_DSA", "0"),
        ("STRATEGY_ND_DENSE", "0"),
        ("DSA_ADJUDICATION", "1"),
        ("DECODE_STABLEHLO_SHA", "0" * 64),
    ],
)
def test_new_recipe_refuses_geometry_configuration_and_pin_drift(field, value):
    env = launch.numerical_environment(profile=PROFILE)
    env["GLM_GREENFIELD_WS32_" + field] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


@pytest.mark.parametrize(
    "change",
    [
        {
            "plan": {
                "prompt_length": 2034,
                "block_rows": 128,
                "context_capacity": 8192,
                "mlp_window": 1,
            }
        },
        {
            "program_options": {
                "paired_position_sort": True,
                "rolled_prefix": False,
                "expert_panels": True,
                "sorted_local_merge": True,
                "key_tile": 512,
            }
        },
        {"model_source_pin": "a" * 40},
        {"evidence": {}},
        {"memory": {}},
        {"graphs": {}},
    ],
)
def test_registration_cannot_change_recipe_or_prerequisites(change, monkeypatch):
    record = a.rolled_registration(ROOT)
    read = Path.read_text

    def changed(path, *args, **kwargs):
        if path == ROOT / a.ROLLED_REGISTRATION:
            return json.dumps({**record, **change})
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError):
        a.rolled_registration(ROOT)


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_memory_caps_charge_growth_but_never_treat_layer_peak_as_model(graph):
    original = a.short_acquisition(ROOT)["fleet"][0]["compiled"][graph]["memory"]
    admitted = {
        **original,
        "argument_size_in_bytes": original["argument_size_in_bytes"] + 4096,
        "temp_size_in_bytes": 1 << 30,
        "generated_code_size_in_bytes": 256 << 20,
    }
    a.validate_short_compiled_memory(graph, admitted, profile=PROFILE, repo=ROOT)
    for key in admitted:
        for value in (float(admitted[key]), admitted[key] + 1, -1):
            with pytest.raises(ValueError):
                a.validate_short_compiled_memory(
                    graph, {**admitted, key: value}, profile=PROFILE, repo=ROOT
                )


def test_registration_does_not_enable_incomplete_whole_model_hlo_guard():
    with pytest.raises(ValueError, match="not integrated"):
        a.authorize_short_graph(
            dict(passed=True, violations=[], block_rows=128), profile=PROFILE, repo=ROOT
        )
    # Historical profile identities and option defaults remain unchanged.
    assert "mlp_window" not in a.short_plan(a.SHORT_PROFILE).identity()
    assert "batched_prefill_program_options" not in a.short_numerical_identity()
    assert a.short_plan(a.PAIRED_SHORT_PROFILE).split == (119, 11)
    options = a.short_program_options(a.PAIRED_SHORT_PROFILE)
    assert options == dict(
        paired_position_sort=True,
        rolled_prefix=False,
        expert_panels=False,
        sorted_local_merge=False,
        key_tile=4096,
    )
    options["key_tile"] = 1
    assert a.short_program_options(a.PAIRED_SHORT_PROFILE)["key_tile"] == 4096


def test_fresh_marker_request_keeps_all_fourteen_pins(monkeypatch):
    env = launch.numerical_environment(profile=PROFILE)
    captured = []
    original = launch.require_short_numerical_request

    def capture(args, **kwargs):
        original(args, **kwargs)
        captured.append(args)

    monkeypatch.setattr(launch, "require_short_numerical_request", capture)
    launch.validate_environment(env)
    args = captured[0]
    a.require_hlo_pin_request(args, compile_only=False, repo=ROOT)
    for graph in a.short_acquisition(ROOT)["graphs"]:
        for form in ("stablehlo_sha256", "optimized_hlo_sha256"):
            bad = deepcopy(args)
            setattr(bad, f"expected_{graph}_{form}", "b" * 64)
            with pytest.raises(ValueError):
                a.require_hlo_pin_request(bad, compile_only=False, repo=ROOT)
