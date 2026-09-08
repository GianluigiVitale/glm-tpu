"""Exercise real parsers/startup, stopping before any runtime or payload work."""

import sys

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import run_short_decoder_ws32 as worker
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_batched_launch as launch


class StartupReached(Exception):
    pass


def cli(tmp_path, profile, target):
    recipe = launch.numerical_environment(profile=profile)
    values = {
        "topology-capture-root": tmp_path / "topology",
        "token-oracle-dir": tmp_path / "tokens",
        "dsa-oracle-dir": tmp_path / "dsa",
        "checkpoint-manifest-sha256": "a" * 64,
        "checkpoint-success-sha256": "b" * 64,
        "token-oracle-manifest-sha256": "c" * 64,
        "token-oracle-success-sha256": "d" * 64,
        "dsa-oracle-manifest-sha256": "e" * 64,
        "dsa-oracle-success-sha256": "f" * 64,
        "dsa-association-summary-sha256": sealer._DSA_ASSOCIATION_SUMMARY_SHA256,
        "dsa-association-success-sha256": sealer._DSA_ASSOCIATION_SUCCESS_SHA256,
        "topology-sha256": "1" * 64,
        "topology-fleet-sha256": "2" * 64,
        "mesh-sha256": "3" * 64,
        "prefill-mode": recipe["GLM_GREENFIELD_WS32_PREFILL_MODE"],
        "prefill-chunk": recipe["GLM_GREENFIELD_WS32_PREFILL_CHUNK"],
        "context-capacity": recipe["GLM_GREENFIELD_WS32_CONTEXT_CAPACITY"],
        "batched-prefill-profile": recipe[
            "GLM_GREENFIELD_WS32_BATCHED_PREFILL_PROFILE"
        ],
        "prefill-memory-reserve-bytes": admission.SHORT_RESERVE_BYTES,
        "prefill-budget-seconds": admission.SHORT_BUDGET_SECONDS,
        "exact-dsa": 1,
        "host-main-rope-table": 1,
        "strategy-nd-dense": 1,
        "strategy-nd-dense-overlay-manifest-sha256": "4" * 64,
        "strategy-nd-dense-overlay-manifest-file-sha256": "5" * 64,
        "strategy-nd-dense-overlay-success-file-sha256": "6" * 64,
        "observer-steps": 14,
        "warmup": 2,
        "iterations": 10,
        "trace-steps": 2,
        "output": tmp_path / "output.json",
    }
    for graph, pins in admission.short_acquisition(launch.REPO, profile=profile)[
        "graphs"
    ].items():
        for form in pins:
            values[
                "expected-" + graph.replace("_", "-") + "-" + form.replace("_", "-")
            ] = recipe[launch.graph_environment_name(graph, form)]
    if target == "worker":
        values.update(
            {
                "coordinator-address": "localhost:12345",
                "num-processes": 8,
                "process-id": 0,
                "slice-name": "db-v4-64-od",
                "checkpoint-root": tmp_path / "checkpoint",
                "source-inventory": tmp_path / "inventory",
                "expected-code-hash": "7" * 40,
                "tensor-output": tmp_path / "output.npz",
                "hlo-dir": tmp_path / "hlo",
                "trace-dir": tmp_path / "trace",
                "strategy-nd-dense-overlay-root": tmp_path / "overlay",
            }
        )
    else:
        values.update(
            {
                "run-dir": tmp_path,
                "code-hash": "7" * 40,
                "source-inventory-sha256": "8" * 64,
                "mode": "numerical",
                "context-label": "2k",
                "tag": "greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1"
                + ("_ps1" if admission.profile_is_paired(profile) else "")
                + "_20260908T220000000000000Z",
            }
        )
    return values


@pytest.mark.parametrize("target", ["worker", "sealer"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("mutation", [None, "decode_vacancy", "wrong_main_stable"])
def test_actual_startup_recipe_reaches_post_pin_sentinel(
    tmp_path, monkeypatch, target, paired, mutation
):
    profile = admission.PAIRED_SHORT_PROFILE if paired else admission.SHORT_PROFILE
    # Source pin checks have dedicated real-source tests; permit the dirty
    # test worktree here. All actual graph request/generic pin checks execute.
    monkeypatch.setattr(
        admission, "require_acquired_model_source", lambda *a, **k: None
    )
    values = cli(tmp_path, profile, target)
    if mutation == "decode_vacancy":
        values["expected-decode-optimized-hlo-sha256"] = "0" * 64
    elif mutation == "wrong_main_stable":
        values["expected-prefill-chunk-stablehlo-sha256"] = "9" * 64
    argv = ["startup-test"] + (["validate"] if target == "sealer" else [])
    for key, value in values.items():
        argv.extend(["--" + key, str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    reached = []

    def stop(*args, **kwargs):
        reached.append(True)
        raise StartupReached

    monkeypatch.setattr(worker, "_require_clean_code", stop)
    monkeypatch.setattr(sealer, "_digest_file", stop)
    if target == "worker":
        invoke = worker.main  # Calls real parse_args, not a fabricated Namespace.
    else:
        invoke = lambda: sealer._validate(sealer._args())
    if mutation is None:
        with pytest.raises(StartupReached):
            invoke()
        assert reached == [True]
    else:
        with pytest.raises((ValueError, SystemExit)):
            invoke()
        assert reached == []
    assert list(tmp_path.iterdir()) == []
