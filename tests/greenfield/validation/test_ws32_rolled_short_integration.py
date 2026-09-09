"""Actual saved TPU graphs through worker publication and independent sealing.

Compiler wrappers expose retained text, not new compilation or numerical output.
No TPU backend, checkpoint payload or performance assertion is involved.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import runpy
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import UNREGISTERED
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill import CHECKS
from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_batched_prefill_runner as adapter

ROOT = Path(__file__).resolve().parents[3]
PROFILE = a.ROLLED_SHORT_PROFILE


def test_actual_shell_prefix_and_rolled_tag():
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    prefix = source.split("readonly DSA_ASSOCIATION_URI=", 1)[0]
    env = {
        "PATH": "/usr/bin:/bin",
        "JAX_PLATFORMS": "cpu",
        **launch.numerical_environment(profile=PROFILE),
    }
    result = subprocess.run(
        ["bash", "-c", prefix + '\nprintf "%s" "$BATCHED_NUMERICAL_CLI"'],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout) == [
        "--batched-prefill-profile",
        PROFILE,
        "--prefill-memory-reserve-bytes",
        str(a.SHORT_RESERVE_BYTES),
    ]
    bad = {**env, "GLM_GREENFIELD_WS32_PREFILL_CHUNK": "17"}
    result = subprocess.run(
        ["bash", "-c", prefix], env=bad, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 2 and "registered main/tail" in result.stderr
    tag = "greenfield_ws32_short_decoder_2k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_20260909T080000000000000Z"
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
        sealer._validate_run_tag(tag.replace("_rp1_ep1_lm1", ""), **options)


def test_real_adapter_sixteen_calls_through_sealer_accounting(monkeypatch):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_batched_prefill_runner.py")
    )
    args, calls, progress = helpers["fake_workload"](
        monkeypatch, prompt=2034, mlp_window=True
    )
    _, token, execution = adapter.execute_graph_pair(
        *args,
        budget_seconds=a.SHORT_BUDGET_SECONDS,
        progress=progress.append,
        fleet_all=bool,
    )
    assert calls == [("prefill_chunk", i * 128, (i + 1) * 128) for i in range(15)] + [
        ("prefill_tail", 1920, 2034)
    ]
    record = dict(
        batched_prefill_profile=PROFILE,
        prefill_mode=worker.PREFILL_MODE,
        batched_prefill_plan=a.short_plan(PROFILE).identity(),
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
    with pytest.raises(ValueError):
        sealer._require_batched_execution(
            {**record, "observed_generated_token_ids": [999]}, **options
        )


@pytest.mark.parametrize(
    "failure",
    [None, "compiled_memory", "peer_preflight", "execution", "reserve", "token"],
)
def test_rolled_actual_worker_memory_and_dispatch_boundary(
    tmp_path, monkeypatch, failure
):
    helpers = runpy.run_path(
        str(ROOT / "tests/greenfield/validation/test_ws32_batched_numerical_worker.py")
    )
    values, calls, votes = helpers["workload"](tmp_path, monkeypatch, failure=failure)
    args = values["args"]
    args.batched_prefill_profile = PROFILE
    values["plan"] = a.short_plan(PROFILE)
    for graph, pins in a.short_acquisition(ROOT, profile=PROFILE)["graphs"].items():
        for form, value in pins.items():
            setattr(args, f"expected_{graph}_{form}", value)
    receipt = json.loads(
        (
            ROOT
            / "docs/artifacts/prefill-rolled-model-compile-db602-sealed-20260909.json"
        ).read_text()
    )
    for graph, record in receipt["programs"].items():
        fields = dict(record["compiled_memory"])
        if failure == "compiled_memory" and graph == "prefill_chunk":
            fields["temp_size_in_bytes"] = (1 << 30) + 1
        values["compiled"][graph] = SimpleNamespace(
            memory_analysis=lambda f=fields: SimpleNamespace(**f)
        )
    if failure is None:
        worker._execute_batched_prefill(**values)
        assert len(calls) == 1 and votes == [True, True]
        preflight = json.loads(
            (tmp_path / "batched_prefill_preflight.rank0.json").read_text()
        )
        assert preflight["plan"] == a.short_plan(PROFILE).identity()
    else:
        with pytest.raises(RuntimeError):
            worker._execute_batched_prefill(**values)
        assert len(calls) == (
            0 if failure in ("compiled_memory", "peer_preflight") else 1
        )
        assert not (tmp_path / "batched_prefill_complete.rank0.json").exists()


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_original_worker_json_and_independent_sealer(graph, tmp_path):
    receipt = json.loads(
        (
            ROOT
            / "docs/artifacts/prefill-rolled-model-compile-db602-sealed-20260909.json"
        ).read_text()
    )
    directory = Path("/home/gianl/glm-run") / receipt["tag"] / "fleet/rank0"
    paths = [
        directory / f"{graph}.{suffix}"
        for suffix in ("stablehlo.mlir", "optimized_hlo.txt")
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("original DB602 compiler evidence unavailable")
    stable, optimized = (p.read_text() for p in paths)
    original = receipt["programs"][graph]
    assert sha256(stable.encode()).hexdigest() == original["stablehlo_sha256"]
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
        block_rows=original["rows"],
        batched_profile=PROFILE,
    )
    assert (actual_stable, actual_optimized) == (stable, optimized)
    assert (tmp_path / f"{graph}.optimized_hlo.txt").read_text() == optimized
    assert report["passed"] and report["profile_registered"]
    assert not report["runtime_memory_admitted"]
    assert not report["numerical_claim"] and not report["performance_claim"]
    assert report["source_location_identity"][
        "program_options"
    ] == a.short_program_options(PROFILE)
    assert (
        report["source_location_identity"]["plan"] == a.short_plan(PROFILE).identity()
    )
    worker._atomic_json(tmp_path / "graph.json", report)
    serialized = json.loads((tmp_path / "graph.json").read_text())
    args = SimpleNamespace(
        batched_prefill_profile=PROFILE,
        **{f"expected_{graph}_{key}": value for key, value in pins.items()},
    )
    replay = sealer._replay_batched_graph(stable, optimized, graph=graph, args=args)
    assert sealer._same(replay, serialized)
    a.validate_short_compiled_memory(
        graph, original["compiled_memory"], profile=PROFILE, repo=ROOT
    )

    # Admission may remove only its registration sentinel, never a failed proof.
    pending = {
        **report,
        "profile_registered": False,
        "passed": False,
        "violations": [UNREGISTERED],
    }
    for name, _ in CHECKS:
        bad = deepcopy(pending)
        bad[name]["passed"] = False
        with pytest.raises(ValueError, match="structural obligations"):
            a.authorize_short_graph(bad, profile=PROFILE, repo=ROOT)
    for key, value in a.short_program_options(PROFILE).items():
        bad = deepcopy(pending)
        bad[key] = not value if type(value) is bool else value + 1
        with pytest.raises(ValueError):
            a.authorize_short_graph(bad, profile=PROFILE, repo=ROOT)
    for field, value in (("program_options", {}), ("plan", {}), ("graph", "decode")):
        bad = deepcopy(pending)
        bad["source_location_identity"][field] = value
        with pytest.raises(ValueError):
            a.authorize_short_graph(bad, profile=PROFILE, repo=ROOT)
    for key in ("stablehlo_sha256", "optimized_hlo_sha256"):
        with pytest.raises(ValueError):
            a.authorize_short_graph(
                {**pending, key: "0" * 64}, profile=PROFILE, repo=ROOT
            )


def test_worker_rejects_wrong_rows_before_inspection(tmp_path):
    with pytest.raises(ValueError, match="rows contradict"):
        worker._write_graph(
            graph="prefill_chunk",
            lowered=SimpleNamespace(
                compiler_ir=lambda **kwargs: "retained compiler fixture"
            ),
            compiled=SimpleNamespace(as_text=lambda: "retained compiler fixture"),
            hlo_dir=tmp_path,
            expected_stable="0" * 64,
            expected_optimized=a.FRESH_OPTIMIZED_MARKER,
            hidden_size=6144,
            exact_dsa=True,
            strategy_nd_dense=True,
            prefill_mode=worker.PREFILL_MODE,
            block_rows=17,
            batched_profile=PROFILE,
        )
