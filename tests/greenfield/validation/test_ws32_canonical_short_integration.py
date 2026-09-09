"""Corrective2K composed admission with DB609 originals, no TPU dispatch."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation import ws32_canonical_prefill_admission as c
from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_batched_prefill_runner as adapter

ROOT = Path(__file__).resolve().parents[3]
PROFILE = c.PROFILE


def test_actual_recipe_source_prerequisites_and_no_historical_inheritance():
    c.require_source(ROOT)
    record = c.registration(ROOT)
    assert record["numerical_inheritance"] is False
    assert record["model_source_pin"] == c.SOURCE_PIN
    identity = a.short_numerical_identity(profile=PROFILE)
    assert identity["frozen_completion_baseline"]["numerical_inheritance"] is False
    assert (
        identity["batched_prefill_acquisition"]["corrected_compiler_receipt_sha256"]
        == c.COMPILER_SHA256
    )
    assert a.short_plan(PROFILE).split == (15, 114)
    assert a.short_context(PROFILE) == "2k"
    assert a.short_budget(PROFILE) == 300
    assert a.short_program_options(PROFILE) == {
        **a.short_program_options(a.ROLLED_SHORT_PROFILE),
        "canonical_dense": True,
    }
    for old in (a.ROLLED_SHORT_PROFILE, a.FROZEN_8K_PROFILE):
        with pytest.raises(ValueError, match="model source differs"):
            a.require_acquired_model_source(ROOT, profile=old)
        assert "canonical_dense" not in a.short_program_options(old)
    with pytest.raises(ValueError, match="not registered"):
        a.short_plan(PROFILE.replace("2k_", "8k_"))
    original = a.short_acquisition(ROOT)["graphs"]
    acquired = a.short_acquisition(ROOT, profile=PROFILE)["graphs"]
    assert len(acquired) == 7
    for graph, pins in acquired.items():
        if graph.startswith("prefill_"):
            assert (
                pins["stablehlo_sha256"] == record["graphs"][graph]["stablehlo_sha256"]
            )
            assert pins["optimized_hlo_sha256"] == a.FRESH_OPTIMIZED_MARKER
        else:
            assert pins == original[graph]


@pytest.mark.parametrize(
    "path", [c.COMPILER_RECEIPT, c.STRUCTURAL_RECEIPT, a.FROZEN_RECEIPT]
)
def test_dependency_bytes_cannot_be_relabelled(path, monkeypatch):
    read = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: read(p) + b" " if p == ROOT / path else read(p)
    )
    with pytest.raises(ValueError, match="evidence drifted"):
        c.registration(ROOT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("SHORT_DECODER_CONTEXT", "8k"),
        ("PREFILL_CHUNK", "32"),
        ("CONTEXT_CAPACITY", "131072"),
        ("EXACT_DSA", "0"),
        ("DSA_ADJUDICATION", "1"),
        ("LATER_EVENT_ALARM_ACK", "1"),
        ("STRATEGY_ND_DENSE", "0"),
        ("HOST_MAIN_ROPE_TABLE", "0"),
        ("DECODE_STABLEHLO_SHA", "0" * 64),
    ],
)
def test_launch_refuses_wrong_recipe(field, value):
    env = launch.numerical_environment(profile=PROFILE)
    env["GLM_GREENFIELD_WS32_" + field] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


def test_actual_shell_prefix_and_distinct_tag():
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    prefix = source.split("readonly DSA_ASSOCIATION_URI=", 1)[0]
    result = subprocess.run(
        ["bash", "-c", prefix],
        env={
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
            **launch.numerical_environment(profile=PROFILE),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    tag = "greenfield_ws32_short_decoder_2k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T220000000000000Z"
    options = dict(
        context_label="2k",
        mode="numerical",
        prefill_chunk=128,
        host_main_rope_table=True,
        prefill_mode=worker.PREFILL_MODE,
        batched_prefill_profile=PROFILE,
    )
    sealer._validate_run_tag(tag, **options)
    with pytest.raises(SystemExit):
        sealer._validate_run_tag(tag.replace("_cd1", ""), **options)


def test_options_reach_both_actual_builders(monkeypatch):
    calls = []
    monkeypatch.setattr(
        adapter,
        "build_ws32_batched_prefill_program",
        lambda *args, **kw: calls.append(kw) or object(),
    )
    adapter.build_graph_pair(
        None,
        SimpleNamespace(context_capacity=8192),
        a.short_plan(PROFILE),
        **a.short_program_options(PROFILE),
    )
    assert [x.pop("block_rows") for x in calls] == [128, 114]
    assert calls == [dict(mlp_window=True, **a.short_program_options(PROFILE))] * 2


def test_actual_shell_geometry_budget_suffix_and_all_fourteen_pin_guard(monkeypatch):
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    section = (
        "# Spec §23.2:"
        + source.split("# Spec §23.2:", 1)[1].split("# The L7 pass criterion", 1)[0]
    )
    env = launch.numerical_environment(profile=PROFILE)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + section
            + '\nprintf "%s %s" "$CHUNK_SUFFIX" "$PREFILL_BUDGET_SECONDS"',
        ],
        env={
            **env,
            "PATH": "/usr/bin:/bin",
            "PREFILL_MODE": worker.PREFILL_MODE,
            "BATCHED_PROFILE": PROFILE,
            "MODE": "numerical",
            "CONTEXT": "2k",
            "CONTEXT_CAPACITY": "8192",
            "HOST_MAIN_ROPE_TABLE": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1 300"
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


def test_actual_sixteen_call_adapter_and_sealer(monkeypatch):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_batched_prefill_runner.py")
    )
    args, calls, progress = helpers["fake_workload"](
        monkeypatch, prompt=2034, mlp_window=True
    )
    _, token, execution = adapter.execute_graph_pair(
        *args,
        budget_seconds=a.short_budget(PROFILE),
        progress=progress.append,
        fleet_all=bool,
    )
    assert len(calls) == 16 and calls[-1] == ("prefill_tail", 1920, 2034)
    record = dict(
        **a.short_numerical_identity(profile=PROFILE),
        prefill_chunk_length=128,
        context_capacity=8192,
        prefill_execution=execution,
        observed_generated_token_ids=[int(token[0])],
    )
    options = dict(mode="numerical", prompt_length=2034, expected_chunk=128)
    sealer._require_batched_execution(json.loads(json.dumps(record)), **options)
    bad = deepcopy(record)
    bad["prefill_execution"]["block_wall_seconds"].pop()
    with pytest.raises(ValueError):
        sealer._require_batched_execution(bad, **options)


@pytest.mark.parametrize(
    "failure",
    [None, "compiled_memory", "peer_preflight", "execution", "reserve", "token"],
)
def test_existing_worker_actual_memory_dispatch_boundary(
    tmp_path, monkeypatch, failure
):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/validation/test_ws32_batched_numerical_worker.py")
    )
    values, calls, votes = helpers["workload"](tmp_path, monkeypatch, failure=failure)
    values["args"].batched_prefill_profile = PROFILE
    values["plan"] = a.short_plan(PROFILE)
    for graph, pins in a.short_acquisition(ROOT, profile=PROFILE)["graphs"].items():
        for form, value in pins.items():
            setattr(values["args"], f"expected_{graph}_{form}", value)
    receipt = json.loads((ROOT / c.COMPILER_RECEIPT).read_text())
    for graph, record in receipt["graphs"].items():
        fields = dict(record["compiled_memory"])
        if failure == "compiled_memory" and graph == "prefill_chunk":
            fields["temp_size_in_bytes"] = (1 << 30) + 1
        values["compiled"][graph] = SimpleNamespace(
            memory_analysis=lambda f=fields: SimpleNamespace(**f)
        )
    if failure is None:
        worker._execute_batched_prefill(**values)
        assert len(calls) == 1 and votes == [True, True]
    else:
        with pytest.raises(RuntimeError):
            worker._execute_batched_prefill(**values)
        assert len(calls) == (
            0 if failure in ("compiled_memory", "peer_preflight") else 1
        )
        assert not (tmp_path / "batched_prefill_complete.rank0.json").exists()


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_db609_worker_publication_and_independent_sealer(graph, tmp_path):
    receipt = json.loads((ROOT / c.COMPILER_RECEIPT).read_text())
    root = Path("/home/gianl/glm-run") / receipt["tag"] / "rank0"
    stable, optimized = (
        (root / f"{graph}.{suffix}").read_text()
        for suffix in ("stablehlo.mlir", "optimized_hlo.txt")
    )
    original = receipt["graphs"][graph]
    assert sha256(optimized.encode()).hexdigest() == original["optimized_hlo_sha256"]
    pins = a.short_acquisition(ROOT, profile=PROFILE)["graphs"][graph]
    report, actual_stable, actual_optimized = worker._write_graph(
        graph=graph,
        lowered=SimpleNamespace(compiler_ir=lambda **kwargs: stable),
        compiled=SimpleNamespace(as_text=lambda: optimized),
        hlo_dir=tmp_path,
        expected_stable=pins["stablehlo_sha256"],
        expected_optimized=pins["optimized_hlo_sha256"],
        hidden_size=6144,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
        prefill_mode=worker.PREFILL_MODE,
        block_rows=128 if graph == "prefill_chunk" else 114,
        batched_profile=PROFILE,
    )
    assert (actual_stable, actual_optimized) == (stable, optimized)
    assert report["passed"] and report["profile_registered"]
    assert report["canonical_dense_proof"]["passed"] and report["canonical_dense"]
    assert not report["runtime_memory_admitted"] and not report["numerical_claim"]
    worker._atomic_json(tmp_path / "graph.json", report)
    serialized = json.loads((tmp_path / "graph.json").read_text())
    args = SimpleNamespace(
        batched_prefill_profile=PROFILE,
        **{f"expected_{graph}_{k}": v for k, v in pins.items()},
    )
    replay = sealer._replay_batched_graph(stable, optimized, graph=graph, args=args)
    assert sealer._same(replay, serialized)
    a.validate_short_compiled_memory(
        graph, original["compiled_memory"], profile=PROFILE, repo=ROOT
    )
    for name in (
        "canonical_dense_proof",
        "rolled_transition_proof",
        "operand_health_proof",
    ):
        # Reuse the actual complete report; mutation must fail even when its
        # top-level verdict is rewritten to look like unregistered success.
        bad = deepcopy(report)
        bad.update(
            passed=False,
            profile_registered=False,
            violations=["batched prefill profile is not registered"],
        )
        from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import UNREGISTERED

        bad["violations"] = [UNREGISTERED]
        bad[name]["passed"] = False
        with pytest.raises(ValueError, match="structural obligations"):
            a.authorize_short_graph(bad, profile=PROFILE, repo=ROOT)


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_actual_compiler_memory_schema_and_all_caps(graph):
    memory = json.loads((ROOT / c.COMPILER_RECEIPT).read_text())["graphs"][graph][
        "compiled_memory"
    ]
    a.validate_short_compiled_memory(graph, memory, profile=PROFILE, repo=ROOT)
    for key, value in {
        "argument_size_in_bytes": memory["argument_size_in_bytes"] + (1 << 20),
        "temp_size_in_bytes": (1 << 30) + 1,
        "generated_code_size_in_bytes": (256 << 20) + 1,
        "output_size_in_bytes": memory["output_size_in_bytes"] + 1,
        "alias_size_in_bytes": 1,
    }.items():
        with pytest.raises(ValueError):
            a.validate_short_compiled_memory(
                graph, {**memory, key: value}, profile=PROFILE, repo=ROOT
            )
