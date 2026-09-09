"""Actual entry/continuation lifecycle with explicitly fixture model and memory."""

import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace as NS

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_prefill_frontier_entry as entry
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import seal_short_decoder_ws32 as sealer

ROOT = Path(__file__).resolve().parents[3]
PROFILE = admission.FROZEN_FIRST_WINDOW_PROFILE


def test_profile_reuses_frozen_graphs_and_refuses_promotion():
    assert admission.short_acquisition(ROOT, profile=PROFILE) == admission.short_acquisition(ROOT, profile=admission.FROZEN_8K_PROFILE)
    assert admission.short_program_options(PROFILE) == admission.short_program_options(admission.FROZEN_8K_PROFILE)
    admission.require_acquired_model_source(ROOT, profile=PROFILE)
    identity = admission.short_numerical_identity(profile=PROFILE)
    assert identity["first_window_diagnostic"]["live_counts"] == [128, 32, 32, 32, 32]
    assert identity["first_window_diagnostic"]["compiled_graphs"] == ["exact_materialize", "exact_promote", "prefill_chunk"]
    assert admission.short_budget(PROFILE) == 300
    with pytest.raises(SystemExit, match="diagnostic cannot seal"):
        sealer._validate(NS(prefill_mode="layer_major_raw_v1", batched_prefill_profile=PROFILE))


@pytest.mark.parametrize("mutation", [None, "decode_vacancy", "wrong_main_stable"])
def test_actual_startup(tmp_path, monkeypatch, mutation):
    helpers = runpy.run_path(str(ROOT / "tests/greenfield/runtime/test_ws32_short_actual_startup.py"))
    helpers["test_actual_startup_recipe_reaches_post_pin_sentinel"](
        tmp_path, monkeypatch, "worker", PROFILE, mutation)


def test_actual_shell_profile_and_upload_injection():
    env = {"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", **launch.numerical_environment(profile=PROFILE)}
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    checked = subprocess.run(["bash", "-c", source.split("readonly DSA_ASSOCIATION_URI=", 1)[0]],
                             env=env, capture_output=True, text=True, timeout=20)
    assert checked.returncode == 0, checked.stderr
    injection = source.split('if [[ $BATCHED_PROFILE == ws32_b128_8k_cap8192_first128_diagnostic_v1 ]]; then', 1)[1].split('launch_rc=0', 1)[0]
    script = "BATCHED_PROFILE=" + PROFILE + "\nexecute_command='upload(){ local rc=0; echo original; }; upload'\nif [[ $BATCHED_PROFILE == " + PROFILE + " ]]; then" + injection + '\nprintf "%s" "$execute_command"\n'
    generated = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert "ws32_prefill_frontier_publish.py" in generated
    assert "timeout --signal=TERM --kill-after=30 300" in generated
    assert generated.count("upload(){") == 1 and "echo original" in generated
    syntax = subprocess.run(["bash", "-n"], input=generated, capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize("failure", [None, "post_memory", "inventory", "peer_entry"])
def test_actual_entry_to_five_calls_and_publication(tmp_path, monkeypatch, failure):
    helpers = runpy.run_path(str(ROOT / "tests/greenfield/validation/test_ws32_prefill_frontier_worker.py"))
    calls, config, events = helpers["setup"](tmp_path, monkeypatch, fail=failure)
    calls.journal.close = lambda: events.append(("close",))
    monkeypatch.setattr(entry, "require_acquired_model_source", lambda *a, **k: None)
    validated = []
    monkeypatch.setattr(entry, "validate_short_compiled_memory", lambda graph, *a, **k: validated.append(graph))
    args = NS(batched_prefill_profile=PROFILE, expected_code_hash="a"*40, process_id=0,
              output=tmp_path / "runner.rank0.json")
    jax = NS(process_index=lambda: 3, local_devices=lambda: [NS(id=d) for d in calls.local_slots])
    ids = list(calls.local_slots) + [100+n for n in range(28)]
    graphs = {g: {"passed": True} for g in ("exact_materialize", "exact_promote", "prefill_chunk")}
    if failure == "inventory":
        graphs["prefill_tail"] = {"passed": True}
    import numpy as np
    def run():
        entry.execute(args=args, repo=ROOT, jax=jax, mesh=None, physical_mesh=NS(flattened_device_ids=ids),
                      config=config, prompt_tokens=np.arange(8155, dtype=np.int32),
                      compiled=calls.programs, graphs=graphs,
                      compiled_memory={g: {} for g in graphs}, identity={}, journal=calls.journal,
                      weights=None, wk=None, rope=None,
                      consensus=(lambda ok: False) if failure == "peer_entry" else calls.consensus)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            run()
    else:
        run()
    dispatches = [e for e in events if e[0] == "dispatch"]
    assert len(dispatches) == (5 if failure is None else 1 if failure == "post_memory" else 0)
    if failure in (None, "post_memory"):
        record = json.loads(args.output.read_text())
        assert record["status"] == ("DIAGNOSTIC_FAILED" if failure else "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION")
        assert (tmp_path / "first_window.rank0" / "wide_final.npz").is_file()
        assert set(validated) == set(graphs)
        assert events[-2:] == [("close",), ("vote", True)]
