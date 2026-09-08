"""Small CPU-only tests for profile routing; no model payload or TPU compile."""

import ast
import json
from pathlib import Path
import re
import shlex
import subprocess

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE, SERIAL_PREFILL_MODE
from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal


ROOT = Path(__file__).resolve().parents[3]


def tag_validator():
    path = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_run_tag"
    )
    env = dict(
        re=re,
        DEFAULT_PREFILL_CHUNK=2048,
        DEFAULT_CONTEXT_CAPACITY=8192,
        SERIAL_PREFILL_MODE=SERIAL_PREFILL_MODE,
        PREFILL_MODE=PREFILL_MODE,
        require_prefill_mode=lambda value: value,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[node.name]


@pytest.mark.parametrize("paired", [False, True])
def test_profile_recipe_tag_journal_and_actual_shell_prefix(
    tmp_path, monkeypatch, paired
):
    profile = admission.PAIRED_SHORT_PROFILE if paired else admission.SHORT_PROFILE
    # Working tree edits are intentional during this unit test; source validation
    # has its own admission tests and is never bypassed by the real launch recipe.
    monkeypatch.setattr(
        admission, "require_acquired_model_source", lambda *a, **k: None
    )
    env = launch.numerical_environment(profile=profile)
    assert env["GLM_GREENFIELD_WS32_BATCHED_PREFILL_PROFILE"] == profile
    for graph, pins in admission.short_acquisition(ROOT, profile=profile)[
        "graphs"
    ].items():
        for form, digest in pins.items():
            assert env[launch.graph_environment_name(graph, form)] == digest
    suffix = "_ps1" if paired else ""
    tag = (
        "greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1"
        + suffix
        + "_20260908T210000000000000Z"
    )
    kwargs = dict(
        context_label="2k",
        mode="numerical",
        prefill_chunk=17,
        host_main_rope_table=True,
        prefill_mode=PREFILL_MODE,
        batched_prefill_profile=profile,
    )
    tag_validator()(tag, **kwargs)
    wrong = tag.replace(
        "_bp1" + suffix + "_", "_bp1" + ("" if paired else "_ps1") + "_"
    )
    with pytest.raises(SystemExit):
        tag_validator()(wrong, **kwargs)
    journal = Ws32NumericalJournal(
        tmp_path / "journal.jsonl",
        {
            **admission.short_numerical_identity(profile=profile),
            "compile_only": False,
        },
    )
    journal.close()
    first = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[0])
    assert first["identity"]["batched_prefill_profile"] == profile
    shell = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    prefix = shell.split("readonly DSA_ASSOCIATION_URI=", 1)[0]
    # The in-process recipe above exercises this validator with only its
    # published-source check mocked. A subprocess cannot inherit that mock;
    # replace exactly this already-tested preflight, not the shell mode guards.
    preflight = 'JAX_PLATFORMS=cpu PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python \\\n        "$WORKTREE/scripts/greenfield/ws32_batched_launch.py" --validate-environment'
    assert prefix.count(preflight) == 1
    prefix = prefix.replace(preflight, "true")
    result = subprocess.run(
        ["bash", "-c", prefix + '\nprintf "%s" "$BATCHED_NUMERICAL_CLI"'],
        env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu", **env},
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--batched-prefill-profile " + profile in result.stdout


@pytest.mark.parametrize("profile", ["", "unknown", None, True])
def test_unknown_profile_cannot_open_numerical_journal(tmp_path, profile):
    with pytest.raises(ValueError):
        Ws32NumericalJournal(
            tmp_path / "journal.jsonl",
            {
                "prefill_mode": PREFILL_MODE,
                "compile_only": False,
                "batched_prefill_profile": profile,
            },
        )
    assert not (tmp_path / "journal.jsonl").exists()


def test_worker_and_sealer_identity_calls_carry_actual_profile():
    for name in ("run_short_decoder_ws32.py", "seal_short_decoder_ws32.py"):
        tree = ast.parse((ROOT / "scripts/greenfield" / name).read_text())
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "short_numerical_identity"
        ]
        assert calls
        assert all(
            any(
                k.arg == "profile"
                and ast.unparse(k.value) == "args.batched_prefill_profile"
                for k in call.keywords
            )
            for call in calls
        )
    tree = ast.parse(
        (ROOT / "scripts/greenfield/run_short_decoder_ws32.py").read_text()
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "build_graph_pair"
    ]
    assert len(calls) == 1
    assert any(
        k.arg == "paired_position_sort" and ast.unparse(k.value) == "paired_sort"
        for k in calls[0].keywords
    )


def test_actual_paired_remote_and_sealer_cli():
    source = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
    profile = admission.PAIRED_SHORT_PROFILE
    env = {
        "PATH": "/usr/bin:/bin",
        "MODE": "numerical",
        "PREFILL_MODE": PREFILL_MODE,
        "BATCHED_NUMERICAL_CLI": f" --batched-prefill-profile {profile} --prefill-memory-reserve-bytes 1073741824",
        "PREFILL_BUDGET_SECONDS": "300",
    }
    assignment = next(
        line for line in source.splitlines() if line.startswith("execute_command=")
    )
    result = subprocess.run(
        ["bash", "-c", assignment + '\nprintf "%s" "$execute_command"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    words = shlex.split(result.stdout)
    assert words.count("--batched-prefill-profile") == 1
    assert words[words.index("--batched-prefill-profile") + 1] == profile
    call = source[
        source.index(
            'seal_python \\\n  "$SEAL_ROOT/scripts/greenfield/seal_short_decoder_ws32.py" validate'
        ) :
    ]
    call = call.split(' >"$RUN_DIR/validate.log"', 1)[0]
    result = subprocess.run(
        ["bash", "-c", 'seal_python(){ printf "%s\\n" "$@"; };\n' + call],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    words = result.stdout.splitlines()
    assert words.count("--batched-prefill-profile") == 1
    assert words[words.index("--batched-prefill-profile") + 1] == profile


def test_db_item_keeps_historical_name_and_separates_paired():
    path = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "item_id" for t in x.targets)
            and ast.unparse(x.value).startswith("'s24_batched_prefill_own_short_'")
            for x in n.body
        )
    )
    for paired in (False, True):
        env = dict(
            summary={
                "prefill_mode": PREFILL_MODE,
                "batched_prefill_profile": (
                    admission.PAIRED_SHORT_PROFILE
                    if paired
                    else admission.SHORT_PROFILE
                ),
            },
            PREFILL_MODE=PREFILL_MODE,
            item_id="gate_d_exact_token_dsa_state_cache",
            note="",
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
        assert (
            env["item_id"]
            == ("paired_sort_" if paired else "")
            + "s24_batched_prefill_own_short_gate_d_exact_token_dsa_state_cache"
        )
