"""Real shell/CLI boundaries without cloud operations or model execution."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.greenfield import ws32_batched_launch as launch
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield import ws32_delivery_phase_transport as transport
from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import run_short_decoder_ws32 as worker

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (ROOT / "scripts/greenfield/run_short_decoder_ws32.sh").read_text()
LABELS = ("128k_d1_0", "128k_d0_0", "128k_d0_05", "128k_d0_95", "256k_e0")


@pytest.mark.parametrize("label", LABELS)
def test_actual_shell_request_and_both_parsers(label, monkeypatch):
    env = {"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu",
           **launch.numerical_environment(profile=runtime.PROFILE, context_label=label)}
    # Everything before PIN selection is local preflight/constant construction.
    # Stop before run directory, leases, network, source deployment or JAX TPU.
    prefix = SOURCE.split('RECOVERY_PIN=$(git', 1)[0]
    assignment = next(line for line in SOURCE.splitlines() if line.startswith("execute_command="))
    seal = SOURCE[SOURCE.index('seal_python \\\n  "$SEAL_ROOT/scripts/greenfield/seal_short_decoder_ws32.py" validate'):]
    seal = seal.split(' >"$RUN_DIR/validate.log"', 1)[0]
    script = prefix + '''
TAG=greenfield_ws32_short_decoder_${CONTEXT}_${MODE}${CHUNK_SUFFIX}_20260912T070000000000000Z
PIN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
RUN_DIR=/fixture/run
SEAL_ROOT=$WORKTREE
PROCESS_ID=0
REMOTE_PREFIX=gs://driftbench-dsv4-uc/results/$TAG
coordinator=127.0.0.1:12345
SEAL_AUTHORIZATION_CLI=''
RECOVERY_PIN=$PIN
'''
    # These fragments expand the original flags but do not execute the command.
    script += '[[ $STORAGE_RESERVE_BYTES == 10737418240 && $LOCAL_EVIDENCE_FLOOR_BYTES == 6442450944 ]]\n'
    script += assignment + '\nprintf "%s\\n" "$TAG" "$WORKER_TIMEOUT_SECONDS" "$execute_command"\n'
    script += 'seal_python(){ printf "%s\\n" "$@"; };\n' + seal
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    tag, timeout = lines[:2]
    assert int(timeout) == (72000 if label == "256k_e0" else 39600)
    transport._identity(tag, "a" * 40, label, 0)
    plan = runtime.programs.long_plan(label)
    sealer._validate_run_tag(tag, context_label=label, mode="numerical", prefill_chunk=128,
        context_capacity=plan.context_capacity, host_main_rope_table=True,
        prefill_mode="layer_major_raw_v1", batched_prefill_profile=runtime.PROFILE)
    words = shlex.split(lines[2])
    start = next(i for i, word in enumerate(words) if word.endswith("run_short_decoder_ws32.py"))
    end = words.index('--trace-dir', start) + 2
    argv = words[start:end]
    argv[argv.index('--process-id') + 1] = '0'
    monkeypatch.setattr(sys, "argv", argv)
    args = worker.parse_args()
    runtime.require_request(args, context_label=label, prompt_length=plan.prompt_length, repo=ROOT)
    monkeypatch.setattr(sys, "argv", lines[3:])
    args = sealer._args()
    runtime.require_request(args, context_label=label, prompt_length=plan.prompt_length, repo=ROOT)


@pytest.mark.parametrize("key,value", [
    ("SHORT_DECODER_MODE", "acquire"), ("PREFILL_CHUNK", "32"),
    ("CONTEXT_CAPACITY", "8192"), ("EXACT_DSA", "0"),
    ("STRATEGY_ND_DENSE", "0"), ("HOST_MAIN_ROPE_TABLE", "0"),
    ("ROTARY_DIAGNOSTIC", "1"), ("DSA_ADJUDICATION", "1"),
    ("LATER_EVENT_ALARM_ACK", "1"), ("DECODE_STABLEHLO_SHA", "0" * 64),
    ("DECODE_OPTIMIZED_HLO_SHA", "a" * 64),
])
def test_long_recipe_refuses_wrong_contract(key, value):
    env = launch.numerical_environment(profile=runtime.PROFILE, context_label=LABELS[0])
    env["GLM_GREENFIELD_WS32_" + key] = value
    with pytest.raises(ValueError):
        launch.validate_environment(env)


def test_long_label_required_and_short_not_reinterpreted():
    with pytest.raises(ValueError): launch.numerical_environment(profile=runtime.PROFILE)
    with pytest.raises(ValueError): launch.numerical_environment(context_label=LABELS[0])
