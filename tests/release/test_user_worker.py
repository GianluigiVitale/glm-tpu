"""CPU-only user worker admission; no credentials, backend, weights or cloud."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu import user_request
from scripts.release import ws32_user_worker as worker

TAG = "greenfield_ws32_user_request_20260914T020000000000000Z"
PIN = "a" * 40


def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "RUN_ROOT", tmp_path)
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    value = user_request.from_token_ids([1, 2, 3], request_id="fixture", seed=42, max_new_tokens=2)
    raw = user_request.canonical(value) + b"\n"
    request = root / "request.json"
    request.write_bytes(raw)
    request.chmod(0o600)
    args = SimpleNamespace(output=root / "runner.rank0.json", process_id=0,
        expected_code_hash=PIN, user_request=request, request_file_sha256=sha256(raw).hexdigest(),
        coordinator_address="192.168.0.37:8476", wall_seconds=3600)
    return args, value


def test_valid_input_and_exact_same_retained_site_recipe(tmp_path, monkeypatch):
    from scripts.greenfield import ws32_native_benchmark_entry as native
    args, value = inputs(tmp_path, monkeypatch)
    assert worker.validate_inputs(args) == value
    actual = worker.site_args(args)
    expected = native.parse_args(["--native-benchmark-request", "unused", "--native-benchmark-protocol", "unused",
        "--protocol-sha256", "unused", "--expected-code-hash", PIN, "--process-id", "0",
        "--coordinator-address", args.coordinator_address, "--output", str(args.output)])
    # Every native site/load argument (not benchmark registration flags) agrees.
    for name, value in vars(expected).items():
        if name not in {"native_benchmark_request", "native_benchmark_protocol", "protocol_sha256"}:
            assert getattr(actual, name) == value, name


@pytest.mark.parametrize("problem", ["hash", "permissions", "directory_permissions", "rank", "pin",
    "namespace", "path", "output", "symlink", "cold_exists", "sessions_exist", "wall", "port", "ip", "schema"])
def test_invalid_inputs_fail_before_runtime(tmp_path, monkeypatch, problem):
    args, _ = inputs(tmp_path, monkeypatch)
    root = args.output.parent
    if problem == "hash": args.request_file_sha256 = "b" * 64
    elif problem == "permissions": args.user_request.chmod(0o644)
    elif problem == "directory_permissions": root.chmod(0o755)
    elif problem == "rank": args.process_id = True
    elif problem == "pin": args.expected_code_hash = "--bad"
    elif problem == "namespace": args.output = tmp_path / "benchmark" / "runner.rank0.json"
    elif problem == "path": args.user_request = root / "not-request.json"
    elif problem == "output": args.output.write_text("original")
    elif problem == "symlink": args.output.symlink_to(root / "missing")
    elif problem == "cold_exists": (root / "native.rank0").mkdir()
    elif problem == "sessions_exist": (root / "sessions.rank0").mkdir()
    elif problem == "wall": args.wall_seconds = 86401
    elif problem == "port": args.coordinator_address = "192.168.0.37:9999"
    elif problem == "ip": args.coordinator_address = "shell-fragment:8476"
    elif problem == "schema":
        args.user_request.write_bytes(b'{"benchmark":true}')
        args.request_file_sha256 = sha256(args.user_request.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        worker.validate_inputs(args)


def test_default_off_precedes_even_input_inspection(monkeypatch):
    monkeypatch.delenv("GLM_GREENFIELD_USER_REQUEST", raising=False)
    monkeypatch.setattr(worker, "validate_inputs", lambda *_: pytest.fail("default-off inspected input"))
    with pytest.raises(ValueError, match="default-off"):
        worker.preflight(None)


def test_user_cold_files_get_original_budget_but_not_benchmark_publication(tmp_path):
    from scripts.greenfield import ws32_native_benchmark_transport as cold
    root = tmp_path / TAG / "native.rank0"
    root.mkdir(parents=True)
    assert cold.native_root(root / "runner.json") == root
    with pytest.raises(ValueError, match="name/size"):
        cold.require_write_size(root / "arbitrary.bin", 10)
    # User evidence is a distinct publication workflow. No widening of the
    # benchmark's existing tag/manifest admission just to make it collect.
    with pytest.raises(ValueError, match="tag/pin/rank"):
        cold._identity(TAG, PIN, 0)


def test_worker_dispatch_keeps_original_owner_script_and_default_off(tmp_path):
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/greenfield/run_short_decoder_ws32.py",
        "--user-request", str(tmp_path / "absent"), "--request-file-sha256", "b"*64,
        "--expected-code-hash", PIN, "--process-id", "0", "--coordinator-address", "192.168.0.37:8476",
        "--wall-seconds", "3600", "--output", str(tmp_path / TAG / "runner.rank0.json")],
        cwd=worker.REPO, env={**os.environ, "JAX_PLATFORMS": "cpu", "GLM_GREENFIELD_USER_REQUEST": "0"},
        capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "user worker is default-off" in result.stderr
    assert not (tmp_path / TAG).exists()


@pytest.mark.parametrize("fail_load", [False, True])
def test_worker_main_wires_actual_host_executor_without_scoring(tmp_path, monkeypatch, fail_load):
    """Fake backend/compiled math; real user executor, store and session loop."""
    import json
    import runpy
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_worker as cold
    from scripts.greenfield import ws32_native_benchmark_requests as requests
    from scripts.greenfield import ws32_native_benchmark_observability as observability
    from scripts.greenfield import ws32_native_benchmark_protocol as protocol

    args, request = inputs(tmp_path, monkeypatch)
    fixture = runpy.run_path(str(worker.REPO / "tests/release/test_user_request.py"))
    loaded, store, obs, calls, admissions = fixture["setup_worker"](monkeypatch, args.output.parent)
    monkeypatch.setattr(worker, "parse_args", lambda _: args)
    monkeypatch.setattr(worker, "preflight", lambda _: request)
    monkeypatch.setattr(worker.os, "umask", lambda _: None)  # never change pytest's process mask
    monkeypatch.setattr(original, "_initialize_runtime", lambda _: (
        SimpleNamespace(process_index=lambda: 0), "mesh", "physical", "topology", "fleet"))
    monkeypatch.setattr(original, "_batched_fleet_all", lambda value: value)

    def load(**kwargs):
        assert kwargs["root"] == args.output.parent / "native.rank0"
        assert kwargs["repo"] == worker.REPO and kwargs["args"] is args
        assert kwargs["preserve_memory"] == store.preserve_memory
        if fail_load:
            raise RuntimeError("fixture loader refusal")
        return loaded

    monkeypatch.setattr(cold, "load_runtime", load)
    monkeypatch.setattr(requests, "RequestStore", lambda root, rank: store)
    monkeypatch.setattr(observability, "NativeObservability", lambda *args: obs)
    monkeypatch.setattr(protocol, "benchmark_registry", lambda: pytest.fail("benchmark registry invoked"))
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs:
                        SimpleNamespace(decode=lambda ids, **kw: str(ids)))
    if fail_load:
        with pytest.raises(RuntimeError, match="loader refusal"):
            worker.main([])
        assert calls == []
    else:
        assert worker.main([]) == 0
        assert [c[0] for c in calls] == ["prefill", "decode"]
        assert admissions == ["before_cache", "cache_ready", "prefill_done"]
    result = json.loads(args.output.read_bytes())
    assert result["complete"] is not fail_load
    assert result["benchmark"] is False and result["protected_result_sealed"] is False
    assert result["request_sha256"] == request["request_sha256"]
    if fail_load:
        assert result["failure_type"] == "RuntimeError"
        assert "fixture loader refusal" not in args.output.read_text()
