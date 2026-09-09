"""Corrected8K joins the existing numerical workflow; no TPU dispatch."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest
import numpy as np

from glm_tpu.greenfield.validation import ws32_canonical_8k_admission as c
from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from glm_tpu.greenfield.validation.ws32_prefill import require_batched_profile
from glm_tpu.greenfield.validation.ws32_short_context import (
    load_ws32_adjudicated_divergence,
)
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_batched_prefill_runner as adapter

ROOT = Path(__file__).resolve().parents[3]
PROFILE = c.PROFILE


def test_actual_recipe_record_source_and_no_numerical_inheritance():
    env = launch.numerical_environment(profile=PROFILE)
    assert env["GLM_GREENFIELD_WS32_DSA_ADJUDICATION"] == "1"
    assert a.short_context(PROFILE) == "8k"
    assert a.short_budget(PROFILE) == 1200
    assert a.short_plan(PROFILE) == a.FROZEN_8K_PLAN
    assert a.short_program_options(PROFILE) == a.short_program_options(
        a.CANONICAL_SHORT_PROFILE
    )
    assert (
        a.short_acquisition(ROOT, profile=PROFILE)["graphs"]
        == a.short_acquisition(ROOT, profile=a.CANONICAL_SHORT_PROFILE)["graphs"]
    )
    identity = a.short_numerical_identity(profile=PROFILE)[
        "batched_prefill_acquisition"
    ]
    assert identity["numerical_inheritance"] is False
    assert identity["adjudication_record_sha256"] == c.RECORD_SHA256
    record = load_ws32_adjudicated_divergence(
        ROOT / c.RECORD, expected_sha256=c.RECORD_SHA256, repository_root=ROOT
    )
    assert (
        record.step,
        record.event_index,
        record.producer_layer_id,
        record.decode_position,
    ) == (0, 1, 1, 8155)
    assert len(record.expected_only) == len(record.observed_only) == 6
    assert "live32_" in record.engine_source_run


@pytest.mark.parametrize(
    "field,value",
    [
        ("DSA_ADJUDICATION", "0"),
        ("SHORT_DECODER_CONTEXT", "2k"),
        ("PREFILL_CHUNK", "32"),
        ("CONTEXT_CAPACITY", "131072"),
        ("HOST_MAIN_ROPE_TABLE", "0"),
        ("LATER_EVENT_ALARM_ACK", "1"),
    ],
)
def test_wrong_launch_and_fresh_alarm_refuse(field, value):
    env = launch.numerical_environment(profile=PROFILE)
    env["GLM_GREENFIELD_WS32_" + field] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


@pytest.mark.parametrize("path", list(c.PREREQUISITES) + [c.RECORD])
def test_prerequisite_bytes_refuse(path, monkeypatch):
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: original(p) + b" " if p == ROOT / path else original(p),
    )
    with pytest.raises(ValueError):
        c.require_prerequisites(ROOT)


def test_record_is_profile_specific_and_exact_path():
    kwargs = dict(
        exact_dsa=True,
        host_main_rope_table=True,
        block_rows=128,
        long_context=None,
        mlp_window=True,
        repo=ROOT,
        adjudication_record=ROOT / c.RECORD,
        adjudication_sha256=c.RECORD_SHA256,
    )
    require_batched_profile("layer_major_raw_v1", profile=PROFILE, **kwargs)
    for old in ("", a.CANONICAL_SHORT_PROFILE, a.FROZEN_8K_PROFILE):
        with pytest.raises(ValueError):
            require_batched_profile("layer_major_raw_v1", profile=old, **kwargs)
    for key, value in [
        ("adjudication_record", None),
        ("adjudication_sha256", "0" * 64),
        (
            "adjudication_record",
            ROOT
            / "docs/artifacts/gate-d-ws32-8k-bprime-adjudicated-divergence-20260906.json",
        ),
    ]:
        with pytest.raises(ValueError):
            require_batched_profile(
                "layer_major_raw_v1", profile=PROFILE, **{**kwargs, key: value}
            )
    with pytest.raises(ValueError):
        require_batched_profile(
            "layer_major_raw_v1", profile=PROFILE, **{**kwargs, "long_context": "e0"}
        )


def test_actual_shell_prefix_record_budget_tag_and_pin_guard(monkeypatch):
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    env = launch.numerical_environment(profile=PROFILE)
    result = subprocess.run(
        ["bash", "-c", source.split("readonly DSA_ASSOCIATION_URI=", 1)[0]],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    section = (
        "# Spec §21.2 first-divergent-event adjudication"
        + source.split("# Spec §21.2 first-divergent-event adjudication", 1)[1].split(
            "# The L7 pass criterion", 1
        )[0]
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + section
            + '\nprintf "%s|%s|%s|%s" "$CHUNK_SUFFIX" "$PREFILL_BUDGET_SECONDS" "$DSA_ADJUDICATION_RECORD_8K" "$DSA_ADJUDICATION_RECORD_8K_SHA"',
        ],
        env={
            **env,
            "PATH": "/usr/bin:/bin",
            "WORKTREE": str(ROOT),
            "PREFILL_MODE": "layer_major_raw_v1",
            "BATCHED_PROFILE": PROFILE,
            "MODE": "numerical",
            "CONTEXT": "8k",
            "CONTEXT_CAPACITY": "8192",
            "HOST_MAIN_ROPE_TABLE": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout
        == "_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1|1200|"
        + str(ROOT / c.RECORD)
        + "|"
        + c.RECORD_SHA256
    )
    sealer._validate_run_tag(
        "greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T230000000000000Z",
        mode="numerical",
        context_label="8k",
        prefill_chunk=128,
        context_capacity=8192,
        host_main_rope_table=True,
        prefill_mode="layer_major_raw_v1",
        batched_prefill_profile=PROFILE,
    )
    captured = []
    original = launch.require_short_numerical_request

    def capture(args, **kwargs):
        original(args, **kwargs)
        captured.append(args)

    monkeypatch.setattr(launch, "require_short_numerical_request", capture)
    launch.validate_environment(env)
    args = captured[0]
    a.require_hlo_pin_request(args, compile_only=False, repo=ROOT)
    for graph, pins in a.short_acquisition(ROOT, profile=PROFILE)["graphs"].items():
        for form in pins:
            bad = deepcopy(args)
            setattr(bad, f"expected_{graph}_{form}", "b" * 64)
            with pytest.raises(ValueError):
                a.require_hlo_pin_request(bad, compile_only=False, repo=ROOT)


def test_actual_64call_adapter_and_sealer(monkeypatch):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_batched_prefill_runner.py")
    )
    args, calls, progress = helpers["fake_workload"](
        monkeypatch, prompt=8155, mlp_window=True, tail_graph_rows=114
    )
    _, token, execution = adapter.execute_graph_pair(
        *args,
        budget_seconds=a.short_budget(PROFILE),
        progress=progress.append,
        fleet_all=bool,
    )
    assert len(calls) == 64 and calls[-1] == ("prefill_tail", 8064, 8155)
    record = dict(
        **a.short_numerical_identity(profile=PROFILE),
        prefill_chunk_length=128,
        context_capacity=8192,
        prefill_execution=execution,
        observed_generated_token_ids=[int(token[0])],
    )
    sealer._require_batched_execution(
        json.loads(json.dumps(record)),
        mode="numerical",
        prompt_length=8155,
        expected_chunk=128,
    )
    record["prefill_execution"]["final_frontier"] = 2034
    with pytest.raises(ValueError):
        sealer._require_batched_execution(
            record, mode="numerical", prompt_length=8155, expected_chunk=128
        )


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_actual_graph_publication_and_sealer(graph, tmp_path):
    # Same original graph pair, distinct8K identity and unchanged twelve proofs.
    helpers = runpy.run_path(
        str(
            ROOT
            / "tests/greenfield/validation/test_ws32_canonical_short_integration.py"
        )
    )
    check = helpers["test_db609_worker_publication_and_independent_sealer"]
    check.__globals__["PROFILE"] = PROFILE
    check(graph, tmp_path)


def test_original_live32_sealer_rederivation_and_mutations():
    from safetensors.numpy import load_file

    source = json.loads((ROOT / c.RECORD).read_text())["engine_source_run"]
    archive = Path("/home/gianl/glm-run") / source / "runner.rank0.npz"
    assert (
        sha256(archive.read_bytes()).hexdigest()
        == "9f72d228af5e71c7a1ad0220a73f51c087c514ac72f62e0d6d460f51ba7cc6be"
    )
    with np.load(archive, allow_pickle=False) as data:
        arrays = dict(data)
    oracle_arrays = load_file(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle/dsa_events.safetensors"
    )
    oracle = SimpleNamespace(
        **{
            k: oracle_arrays[k]
            for k in ("selected_positions", "selected_scores", "valid_counts")
        }
    )
    record = load_ws32_adjudicated_divergence(
        ROOT / c.RECORD, expected_sha256=c.RECORD_SHA256, repository_root=ROOT
    )
    kwargs = dict(oracle=oracle, adjudication=record, repository_root=ROOT, rank=0)
    sealer._rederive_ws32_adjudication(arrays=arrays, **kwargs)
    bad = dict(arrays)
    bad["dsa_selected_scores"] = arrays["dsa_selected_scores"].copy()
    bad["dsa_selected_scores"][0, 1] += 50
    with pytest.raises(SystemExit):
        sealer._rederive_ws32_adjudication(arrays=bad, **kwargs)
    with pytest.raises(SystemExit):
        sealer._rederive_ws32_adjudication(
            arrays=arrays,
            **{
                **kwargs,
                "adjudication": replace(record, expected_only=(1, 2, 3, 4, 5, 6)),
            },
        )


def test_recovery_alarm_environment_does_not_bypass_sealer(tmp_path):
    env = launch.numerical_environment(profile=PROFILE)
    env.update(
        GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_ACK="1",
        GLM_GREENFIELD_WS32_SHORT_DECODER_RECOVER="1",
    )
    launch.validate_environment(env)
    # Reuse unchanged sealer behavioral fixture, not another fake fullmodel.
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/validation/test_ws32_short_sealer.py")
    )
    helpers["test_an_alarm_acknowledgement_must_name_a_pin_this_seal_declares"](
        tmp_path
    )
