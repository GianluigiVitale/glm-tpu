from __future__ import annotations

import ast
import base64
import gzip
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.accepted_compile_only_hlo import (
    ACCEPTED_ENGINE_KWARGS,
    ACCEPTED_FULL_CODE_PIN,
    ACCEPTED_RUNTIME_CONFIG_JSON,
    ACCEPTED_RUNTIME_CONFIG_SHA256,
    ACCEPTED_VLLM_VERSION,
    ACCEPTED_VLLM_VERSION_FILE_SHA256,
    CLAIM_SCOPE,
    validate_compile_only_log,
    validate_hlo_owner_receipts,
    validate_scheduled_hlo_files,
)
from glm_tpu.greenfield.benchmarking.callback_executable_class import (
    ACCEPTED_FINGERPRINTS,
    TOKEN_BUCKETS,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from scripts.greenfield.compile_accepted_db485_hlo import accepted_engine_kwargs
from scripts.greenfield.publish_accepted_db485_compile_only_hlo import (
    canonical_success_bytes,
    expected_payload,
    publish_nonterminal,
    publish_terminal,
    validate_ledger_records,
    validate_remote_names,
    validate_success_fields,
)
from scripts.greenfield.publish_accepted_db485_compile_only_hlo import (
    validate_regular_tree as validate_publish_tree,
)
from scripts.greenfield.seal_accepted_db485_compile_only_hlo import (
    main as seal_main,
)
from scripts.greenfield.seal_accepted_db485_compile_only_hlo import (
    validate_regular_tree as validate_seal_tree,
)


def _log(*, tag: str = "unit_tag", triples=ACCEPTED_FINGERPRINTS) -> bytes:
    pid = 19
    lines = [
        f"ACCEPTED_DB485_COMPILE_ONLY_START tag={tag} code={ACCEPTED_FULL_CODE_PIN} generate_calls=0",
        f"(EngineCore pid={pid}) INFO 08-29 GLM_CODE_FINGERPRINT: git={ACCEPTED_FULL_CODE_PIN[:12]} dirty=0",
        (
            f"(EngineCore pid={pid}) INFO 08-29 Initializing a V1 LLM engine "
            "(v0.1.dev1+ga30addc75) with config: speculative_config=None, "
            "trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8704, "
            "load_format=runai_streamer, tensor_parallel_size=32, "
            "pipeline_parallel_size=1, data_parallel_size=1, "
            "decode_context_parallel_size=1, "
            "served_model_name=gs://driftbench-dsv4-uc/models/GLM-5.2-FP8, "
            "enable_prefix_caching=False, "
            "compilation_config={'debug_dump_path': None}"
        ),
        (
            "ACCEPTED_DB485_RUNTIME_CONFIG "
            f"sha256={ACCEPTED_RUNTIME_CONFIG_SHA256} "
            f"json={ACCEPTED_RUNTIME_CONFIG_JSON}"
        ),
    ]
    for tokens, (executable, including_data, host_transfer) in zip(
        TOKEN_BUCKETS, triples, strict=True
    ):
        lines.extend(
            (
                f"(EngineCore pid={pid}) I0829 (HLO module jit_step_fun_impl): Executable fingerprint:{executable}",
                f"(EngineCore pid={pid}) I0829 (HLO module jit_step_fun_impl): Executable fingerprint (including data segments):{including_data}",
                f"(EngineCore pid={pid}) I0829 (HLO module jit_step_fun_impl): Host transfer fingerprint:{host_transfer}",
                f"(EngineCore pid={pid}) INFO 08-29 Compilation of worker0 backbone --> {{'num_tokens': {tokens}, 'num_reqs': 1}} finished",
            )
        )
    lines.append(f"ACCEPTED_DB485_COMPILE_ONLY_COMPLETE tag={tag} generate_calls=0")
    return ("\n".join(lines) + "\n").encode()


def _hlo(tokens: int, index: int) -> bytes:
    lines = [
        "HloModule jit_step_fun_impl, is_scheduled=true, num_partitions=32",
        f"ENTRY e{index} {{",
    ]
    lines.extend(
        f"  x{line} = bf16[{tokens},6144] all-reduce(p{line}), "
        f'metadata={{op_name="VllmRowParallelLinear/shard_map/psum"}}'
        for line in range(156)
    )
    lines.append("}")
    return ("\n".join(lines) + "\n").encode()


def _hlo_paths(tmp_path: Path) -> list[Path]:
    paths = []
    for index, tokens in enumerate(TOKEN_BUCKETS):
        path = tmp_path / (
            f"jit_step_fun_impl.m{tokens}.module_{100 + index}."
            "jit_step_fun_impl.cl_914450892.after_codegen.txt.gz"
        )
        path.write_bytes(gzip.compress(_hlo(tokens, index), mtime=0))
        paths.append(path)
    return paths


def test_compile_only_log_matches_db485_in_all_three_dimensions() -> None:
    raw = _log()
    report = validate_compile_only_log(raw, run_tag="unit_tag")
    assert report["status"] == "DB485_EXECUTABLE_CLASS_MATCH"
    assert report["claim_scope"] == CLAIM_SCOPE
    assert report["log_sha256"] == sha256(raw).hexdigest()


def test_compile_only_log_refuses_one_fingerprint_change() -> None:
    changed = list(ACCEPTED_FINGERPRINTS)
    changed[0] = ("f" * 64, changed[0][1], changed[0][2])
    with pytest.raises(BenchmarkValidationError, match="differs from DB485"):
        validate_compile_only_log(_log(triples=tuple(changed)), run_tag="unit_tag")


def test_compile_only_log_refuses_reordered_triple_fields() -> None:
    raw = _log()
    lines = raw.decode().splitlines()
    lines[4], lines[5] = lines[5], lines[4]
    with pytest.raises(BenchmarkValidationError, match="triple order"):
        validate_compile_only_log(
            ("\n".join(lines) + "\n").encode(), run_tag="unit_tag"
        )


def test_compile_only_log_refuses_generation_marker() -> None:
    raw = _log() + b"[longctx] correct=True\n"
    with pytest.raises(BenchmarkValidationError, match="generation marker"):
        validate_compile_only_log(raw, run_tag="unit_tag")


def test_compile_only_log_refuses_runtime_configuration_drift() -> None:
    raw = _log().replace(b'"max_model_len":8704', b'"max_model_len":8705')
    with pytest.raises(BenchmarkValidationError, match="runtime configuration"):
        validate_compile_only_log(raw, run_tag="unit_tag")


def test_compile_only_log_refuses_wrong_tag_and_second_owner() -> None:
    raw = _log()
    with pytest.raises(BenchmarkValidationError, match="lifecycle"):
        validate_compile_only_log(raw, run_tag="other_tag")
    second_owner = b"\n".join(
        line.replace(b"pid=19", b"pid=20")
        for line in raw.splitlines()
        if b"pid=19" in line
    )
    hostile = raw + second_owner + b"\n"
    with pytest.raises(BenchmarkValidationError, match="owner count"):
        validate_compile_only_log(hostile, run_tag="unit_tag")


def test_seven_scheduled_hlos_are_required_and_byte_sealed(tmp_path: Path) -> None:
    paths = _hlo_paths(tmp_path)
    report = validate_scheduled_hlo_files(paths)
    assert report["file_count"] == 7
    assert [item["num_tokens"] for item in report["files"]] == list(TOKEN_BUCKETS)
    with pytest.raises(BenchmarkValidationError, match="expected seven"):
        validate_scheduled_hlo_files(paths[:-1])


def test_scheduled_hlo_refuses_duplicate_or_wrong_module(tmp_path: Path) -> None:
    paths = _hlo_paths(tmp_path)
    with pytest.raises(BenchmarkValidationError, match="token-bucket mapping"):
        validate_scheduled_hlo_files(paths[:-1] + [paths[0]])
    paths[1].write_bytes(gzip.compress(b"HloModule helper, is_scheduled=true\n"))
    with pytest.raises(BenchmarkValidationError, match="wrong HLO"):
        validate_scheduled_hlo_files(paths)


def test_scheduled_hlo_refuses_wrong_bucket_shape_and_module_order(
    tmp_path: Path,
) -> None:
    paths = _hlo_paths(tmp_path)
    paths[2].write_bytes(gzip.compress(_hlo(64, 2), mtime=0))
    with pytest.raises(BenchmarkValidationError, match="bucket shape"):
        validate_scheduled_hlo_files(paths)
    paths = _hlo_paths(tmp_path)
    renamed = tmp_path / paths[1].name.replace("module_101", "module_099")
    paths[1].rename(renamed)
    paths[1] = renamed
    with pytest.raises(BenchmarkValidationError, match="module order"):
        validate_scheduled_hlo_files(paths)


def test_hlo_owner_receipts_require_exact_unique_workers() -> None:
    raw = "\n".join(
        ["HLO_OWNER db-v4-64-od-w-3 count=7"]
        + [f"HLO_NONOWNER db-v4-64-od-w-{index}" for index in range(8) if index != 3]
    ).encode()
    report = validate_hlo_owner_receipts(raw)
    assert report["owner_worker"] == 3
    hostile = raw.replace(b"db-v4-64-od-w-7", b"db-v4-64-od-w-6")
    with pytest.raises(BenchmarkValidationError, match="host contract"):
        validate_hlo_owner_receipts(hostile)


def test_driver_kwargs_are_exact_db485_and_have_no_request() -> None:
    kwargs = accepted_engine_kwargs()
    assert kwargs == ACCEPTED_ENGINE_KWARGS
    assert kwargs["max_model_len"] == 8704
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["max_num_batched_tokens"] == 2048
    assert kwargs["num_gpu_blocks_override"] == 24
    assert kwargs["decode_context_parallel_size"] == 1
    assert kwargs["tensor_parallel_size"] == 32
    assert kwargs["load_format"] == "runai_streamer"
    assert "additional_config" not in kwargs
    assert "speculative_config" not in kwargs

    repo = Path(__file__).resolve().parents[3]
    driver = repo / "scripts/greenfield/compile_accepted_db485_hlo.py"
    tree = ast.parse(driver.read_text())
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "generate" not in called_attributes
    source = driver.read_text()
    assert "provenance" not in source
    assert "results.db" not in source


def test_protected_wrapper_is_compile_only_and_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_accepted_db485_compile_only_hlo.sh"
    ).read_text()
    assert "glm_longctx.py" not in wrapper
    assert "results.db" not in wrapper
    assert "raw passkey item" not in wrapper
    assert ".glm_pod_workload.lock" in wrapper
    assert "strict_census pre" in wrapper
    assert "strict_census post" in wrapper
    assert "DB485_EXECUTABLE_CLASS_MATCH" not in wrapper
    assert "compile_accepted_db485_hlo.py" in wrapper
    assert "gate_d_claim=false" in wrapper
    assert "remote_prefix_owned -eq 1" in wrapper
    assert "terminal_publication_started -eq 0" in wrapper
    assert wrapper.index(" nonterminal ") < wrapper.index('>"$RUN_DIR/SUCCESS"')
    assert wrapper.index('>"$RUN_DIR/SUCCESS"') < wrapper.index(" terminal ")
    post_census = wrapper.index("strict_census post")
    assert post_census < wrapper.index("runtime_started=0", post_census)
    assert "DRIVER_SESSION=$driver_pid" in wrapper
    assert "driver_waited -ge 7200" in wrapper
    assert 'kill -TERM -- "-$DRIVER_SESSION"' in wrapper
    assert "bounded 900" in wrapper
    assert "OPTIONS" in wrapper and '",ro,"' in wrapper
    assert 'if [ ! -e "$raw" ]' in wrapper
    assert 'if [ ! -d "$raw" ] || [ ! -r "$raw" ] || [ ! -x "$raw" ]' in wrapper
    assert "actual_vllm_version=$(PYTHONPATH=" in wrapper
    assert 'actual_vllm_version" = "$vllm_version' in wrapper
    assert "! -L $RUN_DIR" in wrapper
    assert "! -L $CODE_BUNDLE_LOCAL" in wrapper
    assert "! -L $VLLM_ARCHIVE_LOCAL" in wrapper
    assert "! -L $VLLM_TAR_LOCAL" in wrapper


def _run_real_compactor(
    wrapper: str, raw: Path, compact: Path
) -> subprocess.CompletedProcess[str]:
    assignment = next(
        line for line in wrapper.splitlines() if line.startswith("compact='")
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            f'HLO_RAW={raw!s}; HLO_COMPACT={compact!s}; {assignment}; eval "$compact"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_real_vacancy(
    wrapper: str, paths: list[Path]
) -> subprocess.CompletedProcess[str]:
    assignment = next(
        line for line in wrapper.splitlines() if line.startswith("vacancy='")
    )
    names = (
        "CODE_RUNTIME",
        "CODE_RUNTIME_TEMP",
        "CODE_BUNDLE_REMOTE",
        "VLLM_RUNTIME",
        "VLLM_ARCHIVE_REMOTE",
        "DUMP_ROOT",
    )
    bindings = "; ".join(
        f"{name}={path!s}" for name, path in zip(names, paths, strict=True)
    )
    return subprocess.run(
        ["bash", "-c", f'{bindings}; {assignment}; eval "$vacancy"'],
        check=False,
        capture_output=True,
        text=True,
    )


def test_real_compactor_distinguishes_absent_empty_and_invalid_roots(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_accepted_db485_compile_only_hlo.sh"
    ).read_text()
    absent = _run_real_compactor(wrapper, tmp_path / "absent", tmp_path / "out1")
    assert absent.returncode == 0
    assert "HLO_NONOWNER" in absent.stdout
    empty = tmp_path / "empty"
    empty.mkdir()
    empty_result = _run_real_compactor(wrapper, empty, tmp_path / "out2")
    assert empty_result.returncode == 0
    assert "HLO_NONOWNER" in empty_result.stdout
    invalid = tmp_path / "not_a_directory"
    invalid.write_text("hostile")
    invalid_result = _run_real_compactor(wrapper, invalid, tmp_path / "out3")
    assert invalid_result.returncode == 0
    assert "HLO_BAD" in invalid_result.stdout
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    unreadable.chmod(0)
    try:
        unreadable_result = _run_real_compactor(wrapper, unreadable, tmp_path / "out4")
        assert unreadable_result.returncode == 0
        assert "HLO_BAD" in unreadable_result.stdout
    finally:
        unreadable.chmod(0o700)


def test_real_fleet_vacancy_refuses_a_dangling_symlink(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_accepted_db485_compile_only_hlo.sh"
    ).read_text()
    paths = [tmp_path / f"target_{index}" for index in range(6)]
    paths[2].symlink_to(tmp_path / "absent_destination")
    result = _run_real_vacancy(wrapper, paths)
    assert result.returncode == 0
    assert "VACANT_BAD" in result.stdout


def test_reconstructed_vllm_transport_imports_exact_accepted_version(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    source = repo / "scripts/greenfield/resources/accepted_vllm_version.py"
    assert sha256(source.read_bytes()).hexdigest() == ACCEPTED_VLLM_VERSION_FILE_SHA256
    archive = tmp_path / "vllm.tar"
    subprocess.run(
        [
            "git",
            "-C",
            "/home/gianl/vllm-build-a30addc",
            "archive",
            "--format=tar",
            "--output",
            str(archive),
            "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
        ],
        check=True,
    )
    with tarfile.open(archive, "a") as stream:
        stream.add(source, arcname="vllm/_version.py")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with tarfile.open(archive) as stream:
        stream.extractall(runtime, filter="data")
    result = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-c",
            (
                "import pathlib,vllm; "
                "pathlib.Path(vllm.__file__).resolve().relative_to("
                f"pathlib.Path({str(runtime)!r}).resolve()); "
                "print(vllm.__version__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(runtime)},
    )
    assert result.stdout.strip() == ACCEPTED_VLLM_VERSION


def test_task_owned_session_can_be_interrupted_without_survivor(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    process = subprocess.Popen(
        [
            "setsid",
            "--wait",
            "bash",
            "-c",
            f"echo $$ > {marker}; exec sleep 60",
        ]
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.01)
        assert os.getsid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_archive_inventory_rejects_extra_and_remote_partial_or_early_terminal(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("sealed\n")
    manifest = {
        "archive_files_before_manifest": [
            {
                "byte_count": payload.stat().st_size,
                "path": "payload.txt",
                "sha256": sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    records = expected_payload(manifest, run)
    expected = {record["path"] for record in records} | {"remote_objects.json"}
    validate_remote_names(expected, expected, terminal_allowed=False)
    with pytest.raises(SystemExit, match="remote object set"):
        validate_remote_names(
            expected, expected - {"payload.txt"}, terminal_allowed=False
        )
    with pytest.raises(SystemExit, match="remote object set"):
        validate_remote_names(expected, expected | {"extra"}, terminal_allowed=False)
    with pytest.raises(SystemExit, match="remote object set"):
        validate_remote_names(expected, expected | {"SUCCESS"}, terminal_allowed=False)
    (run / "unexpected.txt").write_text("not sealed")
    with pytest.raises(SystemExit, match="unsealed"):
        expected_payload(manifest, run)


def test_archive_ledger_replays_every_manifest_byte(tmp_path: Path) -> None:
    import google_crc32c

    run = tmp_path / "run"
    run.mkdir()
    payload_path = run / "payload.txt"
    payload_path.write_text("sealed\n")
    checksum = google_crc32c.Checksum(payload_path.read_bytes())
    payload = [
        {
            "byte_count": payload_path.stat().st_size,
            "path": "payload.txt",
            "sha256": sha256(payload_path.read_bytes()).hexdigest(),
        }
    ]
    ledger = [
        {
            "crc32c": base64.b64encode(checksum.digest()).decode(),
            "generation": "123",
            "path": "payload.txt",
            "sha256": payload[0]["sha256"],
            "size": payload[0]["byte_count"],
        }
    ]
    validate_ledger_records(ledger, payload, run)
    hostile = [dict(ledger[0], sha256="f" * 64)]
    with pytest.raises(SystemExit, match="ledger record drifted"):
        validate_ledger_records(hostile, payload, run)


@pytest.mark.parametrize("validator", [validate_publish_tree, validate_seal_tree])
def test_archive_tree_refuses_links_special_files_and_linked_root(
    tmp_path: Path, validator
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    run = tmp_path / "run"
    run.mkdir()
    linked_file = run / "linked.txt"
    linked_file.symlink_to(outside)
    with pytest.raises(SystemExit, match="unsafe compile-only tree entry"):
        validator(run)
    linked_file.unlink()
    fifo = run / "special.fifo"
    os.mkfifo(fifo)
    with pytest.raises(SystemExit, match="unsafe compile-only tree entry"):
        validator(run)
    fifo.unlink()
    linked_root = tmp_path / "linked_root"
    linked_root.symlink_to(run, target_is_directory=True)
    with pytest.raises(SystemExit, match="not a real directory"):
        validator(linked_root)


def _publication_manifest(payload: Path, *, run_tag: str) -> dict:
    return {
        "archive_files_before_manifest": [
            {
                "byte_count": payload.stat().st_size,
                "path": payload.name,
                "sha256": sha256(payload.read_bytes()).hexdigest(),
            }
        ],
        "artifact_kind": "greenfield_accepted_db485_compile_only_hlo",
        "provenance": {
            "callback_certificate_sha256": "a" * 64,
            "greenfield_pin": "b" * 40,
            "harness_pin": "c" * 40,
            "launcher_sha256": "d" * 64,
            "network_validator_sha256": "e" * 64,
            "vllm_source_pin": "f" * 40,
        },
        "run_tag": run_tag,
        "source_code_pin": "1" * 40,
    }


def _success_contract(
    manifest: dict,
    manifest_path: Path,
    remote: str,
    receipt: dict[str, str],
) -> dict[str, str]:
    provenance = manifest["provenance"]
    return {
        "accepted_code_pin": manifest["source_code_pin"],
        "artifact_kind": manifest["artifact_kind"],
        "callback_certificate_sha256": provenance["callback_certificate_sha256"],
        "db_run_id": "None",
        "gate_d_claim": "false",
        "greenfield_pin": provenance["greenfield_pin"],
        "harness_pin": provenance["harness_pin"],
        "launcher_sha256": provenance["launcher_sha256"],
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "network_validator_sha256": provenance["network_validator_sha256"],
        "numerical_claim": "false",
        "performance_claim": "false",
        "remote_objects_generation": receipt["generation"],
        "remote_objects_sha256": receipt["sha256"],
        "remote_prefix": remote,
        "run_tag": manifest["run_tag"],
        "vllm_pin": provenance["vllm_source_pin"],
    }


def _write_success(path: Path, fields: dict[str, str]) -> None:
    path.write_bytes(canonical_success_bytes(fields))


class _FakeBlob:
    def __init__(self, bucket: _FakeBucket, name: str) -> None:
        self.bucket = bucket
        self.name = name
        self.generation: str | None = None
        self.size: int | None = None
        self.crc32c: str | None = None

    def upload_from_filename(
        self, filename: str, *, if_generation_match: int, checksum: str
    ) -> None:
        assert if_generation_match == 0
        assert checksum == "crc32c"
        if self.bucket.fail_suffix and self.name.endswith(self.bucket.fail_suffix):
            raise RuntimeError("injected upload failure")
        if self.name in self.bucket.objects:
            raise RuntimeError("generation-zero refusal")
        raw = Path(filename).read_bytes()
        self.bucket.next_generation += 1
        generation = str(self.bucket.next_generation)
        import google_crc32c

        digest = base64.b64encode(google_crc32c.value(raw).to_bytes(4, "big")).decode()
        self.bucket.objects[self.name] = (raw, generation, digest)
        self.bucket.upload_order.append(self.name)
        self._load()

    def _load(self) -> None:
        raw, generation, digest = self.bucket.objects[self.name]
        self.generation = generation
        self.size = len(raw)
        self.crc32c = digest

    def reload(self, *, if_generation_match: int | None = None) -> None:
        if self.name not in self.bucket.objects:
            raise RuntimeError("absent object")
        self._load()
        if (
            if_generation_match is not None
            and str(if_generation_match) != self.generation
        ):
            raise RuntimeError("generation mismatch")


class _FakeBucket:
    name = "driftbench-dsv4-uc"

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str, str]] = {}
        self.upload_order: list[str] = []
        self.next_generation = 100
        self.fail_suffix: str | None = None

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)

    def list_blobs(self, *, prefix: str) -> list[_FakeBlob]:
        return [
            self.blob(name) for name in sorted(self.objects) if name.startswith(prefix)
        ]


def test_fake_backend_enforces_generation_zero_and_terminal_last(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("sealed\n")
    manifest = _publication_manifest(payload, run_tag="fake_tag")
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    bucket = _FakeBucket()
    remote = (
        "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
        "accepted_compile_only_hlo/fake_tag"
    )
    receipt = publish_nonterminal(run, remote, storage_bucket=bucket)
    assert not any(name.endswith("/SUCCESS") for name in bucket.objects)
    with pytest.raises(RuntimeError, match="generation-zero"):
        publish_nonterminal(run, remote, storage_bucket=bucket)
    success = _success_contract(manifest, manifest_path, remote, receipt)
    (run / "SUCCESS").write_text(
        "".join(f"{key}={value}\n" for key, value in reversed(success.items()))
    )
    with pytest.raises(SystemExit, match="SUCCESS bytes drifted"):
        publish_terminal(run, remote, storage_bucket=bucket)
    assert not any(name.endswith("/SUCCESS") for name in bucket.objects)
    success["remote_objects_generation"] = "999999"
    _write_success(run / "SUCCESS", success)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        publish_terminal(run, remote, storage_bucket=bucket)
    assert not any(name.endswith("/SUCCESS") for name in bucket.objects)
    success["remote_objects_generation"] = receipt["generation"]
    _write_success(run / "SUCCESS", success)
    publish_terminal(run, remote, storage_bucket=bucket)
    assert bucket.upload_order[-1].endswith("/SUCCESS")


def test_fake_backend_partial_upload_never_publishes_terminal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("sealed\n")
    (run / "manifest.json").write_text(
        json.dumps(_publication_manifest(payload, run_tag="failing_tag"))
    )
    bucket = _FakeBucket()
    bucket.fail_suffix = "/manifest.json"
    remote = (
        "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
        "accepted_compile_only_hlo/failing_tag"
    )
    with pytest.raises(RuntimeError, match="injected upload failure"):
        publish_nonterminal(run, remote, storage_bucket=bucket)
    assert not any(name.endswith("/SUCCESS") for name in bucket.objects)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", "run_tag"),
        ("extra", "unexpected"),
        ("gate_d_claim", "true"),
        ("numerical_claim", "true"),
        ("performance_claim", "true"),
        ("db_run_id", "485"),
        ("remote_prefix", "gs://wrong"),
        ("manifest_sha256", "0" * 64),
        ("harness_pin", "0" * 40),
    ],
)
def test_success_contract_refuses_missing_extra_claim_and_provenance_drift(
    tmp_path: Path, mutation: str, value: str
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("sealed\n")
    manifest = _publication_manifest(payload, run_tag="hostile_tag")
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    remote = (
        "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
        "accepted_compile_only_hlo/hostile_tag"
    )
    receipt = {"generation": "123", "sha256": "9" * 64}
    fields = _success_contract(manifest, manifest_path, remote, receipt)
    if mutation == "missing":
        fields.pop(value)
    elif mutation == "extra":
        fields[value] = "bad"
    else:
        fields[mutation] = value
    with pytest.raises(SystemExit, match="SUCCESS contract drifted"):
        validate_success_fields(
            fields,
            manifest,
            sha256(manifest_path.read_bytes()).hexdigest(),
            remote,
            receipt["generation"],
            receipt["sha256"],
        )


def test_fake_backend_refuses_exact_prefix_folder_marker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = run / "payload.txt"
    payload.write_text("sealed\n")
    (run / "manifest.json").write_text(
        json.dumps(_publication_manifest(payload, run_tag="marker_tag"))
    )
    remote = (
        "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
        "accepted_compile_only_hlo/marker_tag"
    )
    prefix = remote.removeprefix("gs://driftbench-dsv4-uc/") + "/"
    bucket = _FakeBucket()
    bucket.objects[prefix] = (b"", "99", "AAAAAA==")
    with pytest.raises(SystemExit, match="remote object set drifted"):
        publish_nonterminal(run, remote, storage_bucket=bucket)


def test_sealer_binds_provenance_and_complete_raw_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir(parents=True)
    log = run_dir / "compile.log"
    log.write_bytes(_log(tag="sealed_tag"))
    inventory = []
    bucket_map = []
    for index, tokens in enumerate(TOKEN_BUCKETS):
        raw_name = (
            f"module_{100 + index}.jit_step_fun_impl.cl_914450892.after_codegen.txt"
        )
        raw = _hlo(tokens, index)
        sealed_name = f"jit_step_fun_impl.m{tokens}.{raw_name}.gz"
        (hlo_dir / sealed_name).write_bytes(gzip.compress(raw, mtime=0))
        inventory.append(f"raw/{raw_name}\t{len(raw)}")
        bucket_map.append(f"{tokens}\traw/{raw_name}\t{len(raw)}\t{sealed_name}")
    (hlo_dir / "raw_hlo_inventory.txt").write_text("\n".join(inventory) + "\n")
    (hlo_dir / "bucket_hlo_map.tsv").write_text("\n".join(bucket_map) + "\n")
    receipts = run_dir / "hlo_owners.txt"
    receipts.write_text(
        "HLO_OWNER db-v4-64-od-w-0 count=7\n"
        + "".join(f"HLO_NONOWNER db-v4-64-od-w-{index}\n" for index in range(1, 8))
    )
    output = run_dir / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal",
            "--run-tag",
            "sealed_tag",
            "--run-dir",
            str(run_dir),
            "--log",
            str(log),
            "--hlo-dir",
            str(hlo_dir),
            "--hlo-owner-receipts",
            str(receipts),
            "--output",
            str(output),
            "--greenfield-pin",
            "a" * 40,
            "--accepted-bundle-sha256",
            "b" * 64,
            "--vllm-archive-sha256",
            "c" * 64,
            "--vllm-version",
            ACCEPTED_VLLM_VERSION,
            "--vllm-version-file-sha256",
            ACCEPTED_VLLM_VERSION_FILE_SHA256,
            "--launcher-sha256",
            "d" * 64,
            "--network-validator-sha256",
            "e" * 64,
            "--callback-certificate-sha256",
            "6e58bca961c0629799683ccfb75efda195206885e36975e497630ec379076e48",
            "--accepted-source-pin",
            ACCEPTED_FULL_CODE_PIN,
            "--vllm-source-pin",
            "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
            "--harness-pin",
            "3409bf58a758e7bfa398eb92051db701d623e6b9",
            "--remote-prefix",
            (
                "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/"
                "accepted_compile_only_hlo/sealed_tag"
            ),
        ],
    )
    assert seal_main() == 0
    report = json.loads(output.read_text())
    assert report["hlo"]["raw_inventory"]["line_count"] == 7
    assert len(report["hlo"]["bucket_map"]["records"]) == 7
    assert report["hlo_owner"]["owner_worker"] == 0
    assert report["provenance"]["accepted_bundle_sha256"] == "b" * 64
    assert {item["path"] for item in report["archive_files_before_manifest"]} >= {
        "compile.log",
        "hlo/raw_hlo_inventory.txt",
    }
