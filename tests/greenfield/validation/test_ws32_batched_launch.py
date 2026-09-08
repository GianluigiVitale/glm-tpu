"""Exercise the actual shell recipe/flags without cloud or TPU access."""

import json
from pathlib import Path
import shlex
import subprocess

import pytest

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_PROFILE,
    SHORT_BUDGET_SECONDS,
    SHORT_RESERVE_BYTES,
    short_acquisition,
    short_numerical_identity,
)
from scripts.greenfield import ws32_batched_launch as launch


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"


def test_recipe_preserves_retained_checkpoint_and_all_original_graph_pins():
    original = json.loads(
        (ROOT / "configs/greenfield-ws32-batched-acquisition.json").read_text()
    )["environment"]
    env = launch.numerical_environment()
    changed = {k for k in original if original[k] != env[k]}
    assert changed == {"GLM_GREENFIELD_WS32_SHORT_DECODER_MODE"}
    for graph, pins in short_acquisition(ROOT)["graphs"].items():
        for form, digest in pins.items():
            assert env[launch.graph_environment_name(graph, form)] == digest
    launch.validate_environment(env)


@pytest.mark.parametrize(
    "key,value",
    [
        ("SHORT_DECODER_MODE", "acquire"),
        ("SHORT_DECODER_CONTEXT", "8k"),
        ("BATCHED_PREFILL_PROFILE", ""),
        ("PREFILL_CHUNK", "32"),
        ("CONTEXT_CAPACITY", "131072"),
        ("EXACT_DSA", "0"),
        ("STRATEGY_ND_DENSE", "0"),
        ("HOST_MAIN_ROPE_TABLE", "0"),
        ("ROTARY_DIAGNOSTIC", "1"),
        ("DSA_ADJUDICATION", "1"),
        ("LATER_EVENT_ALARM_ACK", "1"),
        ("DECODE_STABLEHLO_SHA", "0" * 64),
    ],
)
def test_recipe_refuses_wrong_scope_or_vacant_pin(key, value):
    env = launch.numerical_environment()
    env["GLM_GREENFIELD_WS32_" + key] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


def test_actual_wrapper_preflight_accepts_recipe_before_external_work():
    # Execute verbatim prefix only: no checkpoint, network, lock or launch code.
    prefix = WRAPPER.read_text().split("readonly DSA_ASSOCIATION_URI=", 1)[0]
    env = {
        "PATH": "/usr/bin:/bin",
        "JAX_PLATFORMS": "cpu",
        **launch.numerical_environment(),
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
        SHORT_PROFILE,
        "--prefill-memory-reserve-bytes",
        str(SHORT_RESERVE_BYTES),
    ]


def test_expanded_actual_remote_worker_and_sealer_receive_fixed_profile():
    source = WRAPPER.read_text()
    assignment = next(
        line for line in source.splitlines() if line.startswith("execute_command=")
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "MODE": "numerical",
        "PREFILL_MODE": "layer_major_raw_v1",
        "BATCHED_NUMERICAL_CLI": f" --batched-prefill-profile {SHORT_PROFILE} --prefill-memory-reserve-bytes {SHORT_RESERVE_BYTES}",
        "PREFILL_BUDGET_SECONDS": str(SHORT_BUDGET_SECONDS),
    }
    result = subprocess.run(
        ["bash", "-c", assignment + '\nprintf "%s" "$execute_command"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    words = shlex.split(result.stdout)
    for flag, value in (
        ("--batched-prefill-profile", SHORT_PROFILE),
        ("--prefill-memory-reserve-bytes", str(SHORT_RESERVE_BYTES)),
        ("--prefill-budget-seconds", str(SHORT_BUDGET_SECONDS)),
    ):
        assert words[words.index(flag) + 1] == value
        assert words.count(flag) == 1
    # Expand actual controller invocation but replace its function with printf.
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
    assert words[words.index("--batched-prefill-profile") + 1] == SHORT_PROFILE
    assert words[words.index("--prefill-memory-reserve-bytes") + 1] == str(
        SHORT_RESERVE_BYTES
    )
    assert words[words.index("--prefill-budget-seconds") + 1] == str(
        SHORT_BUDGET_SECONDS
    )
    assert words.count("--batched-prefill-profile") == 1
    assert words.count("--prefill-memory-reserve-bytes") == 1


@pytest.mark.parametrize(
    "recover,mode,expected",
    [
        ("0", "layer_major_raw_v1", True),
        ("0", "serial_teacher_forced_v1", False),
        ("1", "serial_teacher_forced_v1", True),
    ],
)
def test_actual_materializer_mode_switch(recover, mode, expected):
    source = WRAPPER.read_text()
    fragment = source[source.index("materialize_args=()") :].split("seal_python", 1)[0]
    result = subprocess.run(
        ["bash", "-c", fragment + '\nprintf "%s" "${materialize_args[*]}"'],
        env={"PATH": "/usr/bin:/bin", "RECOVER": recover, "PREFILL_MODE": mode},
        capture_output=True,
        text=True,
        check=True,
    )
    assert ("--allow-failure-diagnostics" in result.stdout) is expected


def test_identity_returns_independent_nested_records():
    one = short_numerical_identity()
    one["batched_prefill_acquisition"]["code_hash"] = "mutated"
    assert (
        short_numerical_identity()["batched_prefill_acquisition"]["code_hash"]
        != "mutated"
    )
