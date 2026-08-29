from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
)
from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import (
    canonicalize_feature2_optimized_hlo,
)
from glm_tpu.greenfield.benchmarking.pp16_feature2_numerical import (
    _bitwise_mismatches,
    compare_feature2_full_width_numerical_capture,
    compare_feature2_numerical_capture,
    validate_feature2_in_process_cleanup,
)
from glm_tpu.greenfield.benchmarking.pp16_feature2_recovery import (
    _arrays_bitwise_equal,
    _metadata_crc32c,
    authenticate_feature2_full_width_rejection,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)

ROOT = Path("/home/gianl/gcs-models")
TOKEN_ORACLE = ROOT / (
    "oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)
DSA_ORACLE = ROOT / (
    "oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
LAYER1 = ROOT / (
    "oracles/greenfield/glm52/dsa_internals/8k/layer1/"
    "greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/"
    "internals.npz"
)
DB529 = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/"
    "inputs/internal"
)
DB550 = ROOT / (
    "results/greenfield_layer0_dense_partial_capture_"
    "20260813T200736889447458Z/dense_partial_capture.npz"
)
PROTECTED_CAPTURE = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z/"
    "result.npz"
)
ACQUIRED_RUN = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_prefill_acquire_20260829T042559840055981Z"
)
FULL_WIDTH_RECOVERY_SOURCE = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_prefill_numerical_20260829T051119686986506Z"
)
SOURCES_AVAILABLE = all(
    path.exists() for path in (TOKEN_ORACLE, DSA_ORACLE, LAYER1, DB529, DB550)
)


def _python_heredoc_after(source: str, marker: str) -> str:
    section = source.split(marker, 1)[1]
    return section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def test_compile_and_numerical_entrypoints_are_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[3]
    shared = (repo / "scripts/greenfield/acquire_pp16_feature2_prefill.py").read_text()
    numerical = (
        repo / "scripts/greenfield/execute_pp16_feature2_prefill.py"
    ).read_text()
    assert "return run_feature2(parse_args(), execute_main=False)" in shared
    assert "return run_feature2(parse_args(), execute_main=True)" in numerical
    assert shared.count("main_compiled(*main_arguments)") == 1
    assert shared.index("feature2 executable canonical HLO drifted") < shared.index(
        "main_compiled(*main_arguments)"
    )
    assert shared.index("_validate_preexecution_memory(memory_after_compile)") < (
        shared.index("main_compiled(*main_arguments)")
    )
    assert "--expected-main-stablehlo-sha256" in numerical
    assert "--expected-main-canonical-hlo-sha256" in numerical
    assert "--expected-main-canonical-hlo-byte-count" in numerical
    assert "--expected-main-stack-frame-reference-count" in numerical
    assert "--expected-jax-version" in numerical
    assert "--expected-jaxlib-version" in numerical
    assert "--expected-libtpu-version" in numerical
    assert "warmup" not in numerical.lower()

    wrapper = (
        repo / "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
    ).read_text()
    assert "GLM_GREENFIELD_PP16_FEATURE2_NUMERICAL:-0" in wrapper
    assert "GLM_GREENFIELD_PP16_FEATURE2_MODE:-off} == execute_once" in wrapper
    assert "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE:-0" in wrapper
    assert "[[ $FULL_WIDTH_ROUNDED_THEN_SLICE == 1 ]]" in wrapper
    assert "== 0 ||" not in wrapper
    assert "runner_variant_args=(--full-width-rounded-then-slice)" in wrapper
    assert '"${runner_variant_args[@]}"' in wrapper
    assert "warmups=0 invocations=1" in wrapper
    assert '--result-npz "$RUN_DIR/result.npz"' in wrapper
    assert "ACQUIRED_CODE_HASH=a2ea1e9439493b0093824d0084bc46c813fc1c33" in wrapper
    assert "ACQUIRED_MAIN_STABLE_SHA=6c1c69d7" in wrapper
    assert "ACQUIRED_MAIN_OPTIMIZED_SHA=a6307a5f" in wrapper
    assert "ACQUIRED_MAIN_CANONICAL_SHA=9e933384" in wrapper
    assert "ACQUIRED_MAIN_CANONICAL_BYTES=6558627" in wrapper
    assert "ACQUIRED_MAIN_STACK_FRAME_REFERENCES=14561" in wrapper
    assert "ACQUIRED_JAX_VERSION=0.10.1" in wrapper
    assert "ACQUIRED_JAXLIB_VERSION=0.10.1" in wrapper
    assert "ACQUIRED_LIBTPU_VERSION=0.0.41" in wrapper
    assert "verify_acquired_hlo_authorization" in wrapper
    assert "validate_feature2_sealed_hlo_archive_identity" in wrapper
    assert "compare_feature2_full_width_numerical_capture" in wrapper
    assert "validate_feature2_in_process_cleanup" in wrapper
    assert "compare_feature2_numerical_capture(" not in wrapper
    assert "full_width_sealed_boundaries_v2" in wrapper
    assert "expected_mismatch_fields" in wrapper
    assert "archive_identity['optimized_hlo_sha256']!=optimized_pin" not in wrapper
    assert "record['bytes_in_use']>4*1024**2" not in wrapper
    assert "post_process_authenticated_zero_work_hosts" in wrapper
    assert "bench/results.db" not in wrapper
    assert "gsutil" not in wrapper
    assert "upload_ledger_no_clobber" in wrapper
    assert "One or more URLs matched no objects." in wrapper
    assert 'verify_final_object_set "$terminal_status"' in wrapper
    assert '--if-generation-match="$generation"' in wrapper
    create = wrapper.index("gcloud storage cp --if-generation-match=0")
    assert wrapper.rindex("terminal_publication_started=1", 0, create) < create
    assert wrapper.rindex("terminal_remote_name=$terminal_status", 0, create) < create
    assert '2>"$RUN_DIR/terminal_create.receipt.stderr"' in wrapper[create:]
    assert '>"$RUN_DIR/terminal_create.stdout"' in wrapper[create:]
    assert 'sync -f "$RUN_DIR/terminal_create.receipt.stderr"' in wrapper[create:]
    assert "[[ -n $terminal_remote_name && -s $receipt ]] || return 1" in wrapper
    assert 'for key in ("crc32c_hash", "crc32c")' in wrapper
    assert "not crc_values" in wrapper
    assert "len(set(crc_values)) != 1" in wrapper
    assert 'record["crc32c_hash"]' not in wrapper
    assert 'after["crc32c_hash"]' not in wrapper
    assert 'gcloud storage cp --no-clobber "$terminal_path"' not in wrapper


def test_feature2_numerical_wrapper_refuses_the_rejected_variant() -> None:
    wrapper = Path(__file__).parents[3] / (
        "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GLM_GREENFIELD_PP16_FEATURE2_NUMERICAL": "1",
            "GLM_GREENFIELD_PP16_FEATURE2_MODE": "execute_once",
            "GLM_GREENFIELD_PP16_FULL_WIDTH_ROUNDED_THEN_SLICE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "admitted successor" in completed.stderr


def test_feature2_numerical_embedded_verifiers_compile_in_exact_scope() -> None:
    wrapper = Path(__file__).parents[3] / (
        "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
    )
    source = wrapper.read_text()
    authorization = _python_heredoc_after(
        source,
        '"$RUN_DIR/acquisition_authorization.json"',
    )
    verifier = _python_heredoc_after(
        source,
        'say "recomputing HLO/source/load claims and exact numerical result without JAX"',
    )
    compile(authorization, "<pp16-feature2-acquisition-authorization>", "exec")
    compile(verifier, "<pp16-feature2-numerical-verifier>", "exec")
    tree = ast.parse(verifier)
    imports = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "glm_tpu.greenfield.benchmarking.pp16_feature2_hlo",
        "validate_feature2_sealed_hlo_archive_identity",
    ) in imports
    assert (
        "glm_tpu.greenfield.benchmarking.pp16_feature2_numerical",
        "compare_feature2_full_width_numerical_capture",
    ) in imports
    assert (
        "glm_tpu.greenfield.benchmarking.pp16_feature2_numerical",
        "validate_feature2_in_process_cleanup",
    ) in imports
    assert "validate_feature2_sealed_hlo_archive_identity(" in verifier
    assert "compare_feature2_full_width_numerical_capture(" in verifier


@pytest.mark.skipif(
    not all(
        (ACQUIRED_RUN / name).is_file()
        for name in ("evidence.sha256", "runner.json", "summary.json", "HLO_ACQUIRED")
    ),
    reason="sealed feature2 acquisition unavailable",
)
def test_feature2_numerical_acquisition_authorization_is_executable(
    tmp_path: Path,
) -> None:
    wrapper = Path(__file__).parents[3] / (
        "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
    )
    authorization = _python_heredoc_after(
        wrapper.read_text(),
        '"$RUN_DIR/acquisition_authorization.json"',
    )
    directory = tmp_path / "acquired"
    directory.mkdir()
    for name in ("evidence.sha256", "runner.json", "summary.json", "HLO_ACQUIRED"):
        shutil.copyfile(ACQUIRED_RUN / name, directory / name)
    remote = (
        "gs://driftbench-dsv4-uc/results/"
        "greenfield_pp16_feature2_prefill_acquire_20260829T042559840055981Z"
    )
    entries = []
    for line in (directory / "evidence.sha256").read_text().splitlines():
        _, relative = line.split(maxsplit=1)
        entries.append(relative.removeprefix("*").removeprefix("./"))
    objects = sorted(
        [
            f"{remote}/evidence.sha256",
            f"{remote}/HLO_ACQUIRED",
            *(f"{remote}/{name}" for name in entries),
        ]
    )
    (directory / "remote_objects.txt").write_text("\n".join(objects) + "\n")
    output = tmp_path / "authorization.json"
    arguments = [
        sys.executable,
        "-c",
        authorization,
        str(directory),
        str(
            Path(__file__).parents[3]
            / "docs/artifacts/pp16-feature2-sealed-hlo-acquisition.json"
        ),
        "9498097422e8bd06e637a6e78360bae8156992777cd5c92294ab9321139025e0",
        "a2ea1e9439493b0093824d0084bc46c813fc1c33",
        "greenfield_pp16_feature2_prefill_acquire_20260829T042559840055981Z",
        remote,
        "7741bef152843992afc23902e5cab20be52dac55722f1da1ff7f017990449036",
        "75bdd75f03fa4bba530f7863ae3b5728094745ea2e3884fb4d0ae72e8767c566",
        "1dda18f3d007c6859911d29d9b1e526c75e37d748ed3d27f61683026e2218c45",
        "b483460ebee19140d5fc30df77fa9851ad5bb080b307c740a0f6baa781b1d271",
        "abec1454910e319be88e72eb8d7e5dbb55b841911990b1f2f76a1795a536fbde",
        "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4",
        "a6307a5f487b0cfcd79712c45ace89753cf0dc54e332b5c24fe9010a36ae3175",
        "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5",
        "6558627",
        "14561",
        str(output),
    ]
    completed = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text())["passed"] is True

    (directory / "remote_objects.txt").write_text("\n".join(objects[:-1]) + "\n")
    rejected = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "remote object set drifted" in rejected.stderr


@pytest.mark.skipif(
    not (ACQUIRED_RUN / "hlo/feature2_main.optimized_hlo.txt").is_file(),
    reason="sealed feature2 acquisition optimized HLO unavailable",
)
def test_feature2_raw_hlo_provenance_may_differ_only_when_canonical_identity_matches() -> (
    None
):
    optimized_path = ACQUIRED_RUN / "hlo/feature2_main.optimized_hlo.txt"
    acquired_raw = optimized_path.read_text()
    acquired_raw_sha = sha256(acquired_raw.encode()).hexdigest()
    assert acquired_raw_sha == (
        "a6307a5f487b0cfcd79712c45ace89753cf0dc54e332b5c24fe9010a36ae3175"
    )

    file_locations_start = acquired_raw.index("FileLocations\n")
    stack_frames_start = acquired_raw.index("StackFrames\n", file_locations_start)
    file_locations = acquired_raw[file_locations_start:stack_frames_start]
    line_match = re.search(r"\bline=([0-9]+)\b", file_locations)
    assert line_match is not None
    original_line = int(line_match.group(1))
    mutated_locations = (
        file_locations[: line_match.start(1)]
        + str(original_line + 1)
        + file_locations[line_match.end(1) :]
    )
    numerical_raw = (
        acquired_raw[:file_locations_start]
        + mutated_locations
        + acquired_raw[stack_frames_start:]
    )

    acquired_canonical, acquired_identity = canonicalize_feature2_optimized_hlo(
        acquired_raw
    )
    numerical_canonical, numerical_identity = canonicalize_feature2_optimized_hlo(
        numerical_raw
    )
    assert sha256(numerical_raw.encode()).hexdigest() != acquired_raw_sha
    assert numerical_canonical == acquired_canonical
    for identity in (acquired_identity, numerical_identity):
        assert identity["sha256"] == (
            "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
        )
        assert identity["byte_count"] == 6_558_627
        assert identity["stripped_stack_frame_references"] == 14_561


def test_feature2_cleanup_accepts_only_bounded_generated_code_residency() -> None:
    prior = validate_feature2_in_process_cleanup(
        [{"bytes_in_use": 72_812_032, "num_allocs": 6}] * 2,
        generated_code_size_bytes=71_058_944,
    )
    assert prior["mode"] == "generated_code_resident_until_process_exit"
    assert prior["residual_over_generated_code_bytes"] == 1_753_088

    successor = validate_feature2_in_process_cleanup(
        [{"bytes_in_use": 66_017_792, "num_allocs": 6}] * 2,
        generated_code_size_bytes=64_264_704,
    )
    assert successor["mode"] == "generated_code_resident_until_process_exit"
    assert successor["residual_over_generated_code_bytes"] == 1_753_088

    released = validate_feature2_in_process_cleanup(
        [{"bytes_in_use": 1_753_088, "num_allocs": 5}] * 2,
        generated_code_size_bytes=64_264_704,
    )
    assert released["mode"] == "released_to_small_baseline"
    assert released["residual_over_generated_code_bytes"] is None

    with pytest.raises(BenchmarkValidationError, match="exceeds generated-code"):
        validate_feature2_in_process_cleanup(
            [
                {
                    "bytes_in_use": 64_264_704 + 4 * 1024**2 + 1,
                    "num_allocs": 7,
                }
            ]
            * 2,
            generated_code_size_bytes=64_264_704,
        )
    with pytest.raises(BenchmarkValidationError, match="asymmetric or negative"):
        validate_feature2_in_process_cleanup(
            [
                {"bytes_in_use": 66_017_792, "num_allocs": 6},
                {"bytes_in_use": 66_017_792, "num_allocs": 7},
            ],
            generated_code_size_bytes=64_264_704,
        )


def test_full_width_recovery_is_default_off_cpu_only_and_compilable() -> None:
    recovery = Path(__file__).parents[3] / (
        "scripts/greenfield/recover_pp16_feature2_full_width_rejection.sh"
    )
    text = recovery.read_text()
    completed = subprocess.run(
        ["bash", str(recovery)],
        env={
            "GLM_GREENFIELD_PP16_FEATURE2_FULL_WIDTH_RECOVER": "0",
            "PATH": os.environ["PATH"],
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "JAX_PLATFORMS=cpu" in text
    assert "TPU_VISIBLE_DEVICES" not in text
    assert "execute_pp16_feature2_prefill.py" not in text
    assert "source_untouched=true" in text
    assert "1787980539108624" in text
    assert "len(expected) != 28" in text
    assert text.count('for key in ("crc32c_hash", "crc32c")') == 3
    assert text.count("not crc_values") == 3
    assert text.count("len(set(crc_values)) != 1") == 3
    assert "not crc_values or len(set(crc_values)) != 1" in text
    assert "authenticate_feature2_full_width_rejection" in text
    assert "--if-generation-match=0" in text
    assert "gcloud compute tpus tpu-vm create" not in text
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", text, flags=re.DOTALL)
    assert len(programs) == 6
    for index, program in enumerate(programs):
        compile(program, f"<pp16-feature2-full-width-recovery-{index}>", "exec")


@pytest.mark.skipif(
    not SOURCES_AVAILABLE or not FULL_WIDTH_RECOVERY_SOURCE.is_dir(),
    reason="protected full-width feature2 rejection unavailable",
)
def test_full_width_recovery_recomputes_the_sealed_rejection() -> None:
    record = authenticate_feature2_full_width_rejection(
        FULL_WIDTH_RECOVERY_SOURCE,
        source_remote=(
            "gs://driftbench-dsv4-uc/results/"
            "greenfield_pp16_feature2_prefill_numerical_"
            "20260829T051119686986506Z"
        ),
        token_oracle_dir=TOKEN_ORACLE,
        dsa_oracle_dir=DSA_ORACLE,
        layer1_internal_reference=LAYER1,
        db529_internal_dir=DB529,
        db550_boundary=DB550,
        half_width_capture=PROTECTED_CAPTURE,
        db518_prompt_cache_dir=(DB529.parent / "prompt_cache"),
        results_db=Path("/home/gianl/glm-tpu/bench/results.db"),
        recovery_tag=(
            "greenfield_pp16_feature2_full_width_recovery_20260829T000000000000000Z"
        ),
    )
    assert record["status"] == "NUMERICAL_REJECTED"
    assert record["comparison"]["exact"] is False
    assert record["normalized_hidden_localization"] == {
        "actual_bits_per_owner": [48422, 48422],
        "expected_bits": 48423,
        "hidden_index": 2795,
        "mismatches_per_owner": [1, 1],
        "owners_bitwise_equal": True,
    }
    assert record["terminal"]["generation"] == "1787980539108624"
    assert record["terminal"]["crc32c"] == "jCuZbg=="
    assert len(record["ledger_entries"]) == 26
    assert record["full_width_vs_half_width"]["bitwise_equal"] is True
    assert record["full_width_vs_half_width"]["common_array_count"] == 11
    assert record["db518_cache_localization"] == {
        "accepted_cache_sha256": (
            "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
        ),
        "actual_bits": 47091,
        "candidate_cache_sha256": (
            "35350ca026c437def5c958b3382ea3aabc97747d3fbdea8205e1aa336cd48d7a"
        ),
        "earliest_hidden_index": 35,
        "earliest_position": 113,
        "expected_bits": 47092,
        "manifest_sha256": (
            "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab"
        ),
        "mismatch_coordinate_count": 71,
        "mismatch_coordinates_sha256": (
            "72dc17d4ec3fa5473f05cd33f34cc2eda522413d8ca73e4b93e81b2556c6aaf3"
        ),
        "mismatch_position_count": 71,
        "maximum_hidden_index": 63,
        "tensor_file_sha256": (
            "36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a"
        ),
    }


def test_feature2_recovery_crc32c_metadata_is_unambiguous() -> None:
    assert _metadata_crc32c({"crc32c": "AAAAAA=="}) == "AAAAAA=="
    assert _metadata_crc32c({"crc32c_hash": "AAAAAA=="}) == "AAAAAA=="
    assert (
        _metadata_crc32c({"crc32c": "AAAAAA==", "crc32c_hash": "AAAAAA=="})
        == "AAAAAA=="
    )
    with pytest.raises(BenchmarkValidationError, match="absent or conflicting"):
        _metadata_crc32c({})
    with pytest.raises(BenchmarkValidationError, match="absent or conflicting"):
        _metadata_crc32c({"crc32c": "AAAAAA==", "crc32c_hash": "BBBBBB=="})


def test_feature2_recovery_array_identity_is_bitwise_for_signed_zero() -> None:
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert np.array_equal(positive_zero, negative_zero)
    assert not _arrays_bitwise_equal(positive_zero, negative_zero)
    assert _arrays_bitwise_equal(positive_zero, positive_zero.copy())


@pytest.mark.parametrize(
    "metadata",
    (
        {"generation": "123", "size": 7},
        {
            "crc32c": "AAAAAA==",
            "crc32c_hash": "BBBBBB==",
            "generation": "123",
            "size": 7,
        },
    ),
)
def test_terminal_verifiers_fail_closed_under_optimized_python(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    repo = Path(__file__).parents[3]
    sources = (
        (
            repo / "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
        ).read_text(),
        (
            repo / "scripts/greenfield/recover_pp16_feature2_full_width_rejection.sh"
        ).read_text(),
    )
    terminal_programs = []
    for source in sources:
        programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", source, flags=re.DOTALL)
        selected = [
            program
            for program in programs
            if "path, receipt" in program and "crc_values = [" in program
        ]
        assert len(selected) == 2
        assert all("assert " not in program for program in selected)
        terminal_programs.extend(selected)

    marker = tmp_path / "NUMERICAL_REJECTED"
    marker.write_bytes(b"payload")
    remote = "gs://driftbench-dsv4-uc/results/hostile/NUMERICAL_REJECTED"
    receipt = tmp_path / "receipt.stderr"
    receipt.write_text(f"Created: {remote}#123\n")
    describe = tmp_path / "describe.json"
    describe.write_text(json.dumps(metadata))
    for program in terminal_programs:
        completed = subprocess.run(
            [
                sys.executable,
                "-O",
                "-c",
                program,
                str(marker),
                str(receipt),
                str(describe),
                remote,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "authentication failed" in completed.stderr


def test_exact_comparator_rejects_signed_zero_and_one_bit_float_drift() -> None:
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert positive_zero[0] == negative_zero[0]
    assert _bitwise_mismatches(positive_zero, negative_zero) == 1

    baseline = np.asarray([1.0], dtype=np.float32)
    one_bit = baseline.view(np.uint32).copy()
    one_bit[0] ^= np.uint32(1)
    assert _bitwise_mismatches(baseline, one_bit.view(np.float32)) == 1


def _capture(path: Path, *, full_width_boundaries: bool = False) -> None:
    from safetensors import safe_open

    from glm_tpu.greenfield.validation.short_context_dsa_oracle import (
        inspect_short_context_dsa_oracle,
    )

    manifest = inspect_short_context_dsa_oracle(DSA_ORACLE)
    with safe_open(
        DSA_ORACLE / manifest["files"]["tensors"]["filename"], framework="np"
    ) as handle:
        positions = np.ascontiguousarray(
            handle.get_tensor("selected_positions")[0, 1][None, :]
        )
        scores = np.ascontiguousarray(
            handle.get_tensor("selected_scores")[0, 1][None, :]
        )
        valid = np.ascontiguousarray(handle.get_tensor("valid_counts")[0, 1][None])
    with np.load(LAYER1, allow_pickle=False) as handle:
        normalized_bits = np.ascontiguousarray(handle["accepted__normalized_hidden"])
        q_a_bits = np.ascontiguousarray(handle["accepted__q_a_state"])
        dsa_query = np.ascontiguousarray(handle["accepted__query"])
        dsa_head_weights = np.ascontiguousarray(handle["accepted__head_weights"])
        key_bits = np.ascontiguousarray(
            np.asarray(handle["accepted__current_key"], dtype=ml_dtypes.bfloat16).view(
                np.uint16
            )
        )
    with np.load(DB550, allow_pickle=False) as handle:
        order = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        _, carried = derive_expected_dense_boundary_bits(
            handle["dense_virtual_partials_bfloat16_bits"],
            handle["post_attention_residual_bfloat16_bits"],
            order,
        )
    layer1_cache = np.zeros((2, 16, 256, 128), dtype=np.uint16)
    layer1_cache[1, 15, 219] = key_bits
    values = {
        "event1_positions": positions,
        "event1_valid_counts": valid,
        "event1_scores": scores,
        "current_carried_halves_bfloat16_bits": np.stack(
            (carried[:, :3072], carried[:, 3072:])
        ),
        "current_attention_query_owners_bfloat16_bits": np.zeros(
            (2, 1, 32, 256), dtype=np.uint16
        ),
        "current_kv_a_bfloat16_bits": np.zeros((1, 576), dtype=np.uint16),
        "layer0_kv_cache_owners_bfloat16_bits": np.zeros(
            (2, 16, 256, 640), dtype=np.uint16
        ),
        "layer0_index_cache_owners_bfloat16_bits": np.zeros(
            (2, 16, 256, 128), dtype=np.uint16
        ),
        "layer1_index_cache_owners_bfloat16_bits": layer1_cache,
        "carried_liveness_digest_owners": np.zeros((2, 2), dtype=np.uint32),
        "contract_valid": np.ones((1,), dtype=np.bool_),
    }
    if full_width_boundaries:
        values.update(
            {
                "current_normalized_hidden_owners_bfloat16_bits": (
                    np.ascontiguousarray(np.broadcast_to(normalized_bits, (2, 1, 6144)))
                ),
                "current_q_a_state_owners_bfloat16_bits": np.ascontiguousarray(
                    np.broadcast_to(q_a_bits, (2, 1, 2048))
                ),
                "current_dsa_query_owners": np.ascontiguousarray(
                    np.broadcast_to(dsa_query, (2, 1, 32, 128))
                ),
                "current_dsa_head_weights_owners": np.ascontiguousarray(
                    np.broadcast_to(dsa_head_weights, (2, 1, 32))
                ),
            }
        )
    np.savez(path, **values)


def _compare(path: Path) -> dict[str, object]:
    return compare_feature2_numerical_capture(
        path,
        token_oracle_dir=TOKEN_ORACLE,
        dsa_oracle_dir=DSA_ORACLE,
        layer1_internal_reference=LAYER1,
        db529_internal_dir=DB529,
        db550_boundary=DB550,
    )


def _compare_full_width(path: Path) -> dict[str, object]:
    return compare_feature2_full_width_numerical_capture(
        path,
        token_oracle_dir=TOKEN_ORACLE,
        dsa_oracle_dir=DSA_ORACLE,
        layer1_internal_reference=LAYER1,
        db529_internal_dir=DB529,
        db550_boundary=DB550,
    )


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
def test_feature2_numerical_comparator_accepts_only_exact_capture(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture)
    report = _compare(capture)
    assert report["exact"] is True
    assert report["status"] == "NUMERICAL_EXACT"
    assert set(report["mismatch_counts"].values()) == {0}

    with np.load(capture, allow_pickle=False) as handle:
        values = {name: np.asarray(handle[name]).copy() for name in handle.files}
    values["event1_positions"][0, 0] ^= np.int32(1)
    np.savez(capture, **values)
    rejected = _compare(capture)
    assert rejected["exact"] is False
    assert rejected["status"] == "NUMERICAL_REJECTED"
    assert rejected["mismatch_counts"]["event1_positions"] == 1


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
def test_feature2_numerical_comparator_rejects_schema_drift(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture)
    with np.load(capture, allow_pickle=False) as handle:
        values = {name: np.asarray(handle[name]).copy() for name in handle.files}
    values.pop("contract_valid")
    np.savez(capture, **values)
    with pytest.raises(BenchmarkValidationError, match="keys drifted"):
        _compare(capture)


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
def test_full_width_comparator_requires_every_sealed_boundary(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture, full_width_boundaries=True)
    report = _compare_full_width(capture)
    assert report["exact"] is True
    assert report["status"] == "NUMERICAL_EXACT"
    assert report["comparison_schema"] == "full_width_sealed_boundaries_v2"
    assert report["sealed_boundary_comparisons_required"] is True
    assert set(report["mismatch_counts"].values()) == {0}
    for name in (
        "layer1_normalized_hidden_bfloat16_bits",
        "layer1_q_a_state_bfloat16_bits",
        "layer1_dsa_query_float32",
        "layer1_dsa_head_weights_float32",
    ):
        assert name in report["mismatch_counts"]

    base = tmp_path / "base.npz"
    _capture(base)
    with pytest.raises(BenchmarkValidationError, match="keys drifted"):
        _compare_full_width(base)
    with pytest.raises(BenchmarkValidationError, match="keys drifted"):
        _compare(capture)


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
@pytest.mark.parametrize(
    ("field", "mismatch_name"),
    (
        (
            "current_normalized_hidden_owners_bfloat16_bits",
            "layer1_normalized_hidden_bfloat16_bits",
        ),
        (
            "current_q_a_state_owners_bfloat16_bits",
            "layer1_q_a_state_bfloat16_bits",
        ),
        ("current_dsa_query_owners", "layer1_dsa_query_float32"),
        (
            "current_dsa_head_weights_owners",
            "layer1_dsa_head_weights_float32",
        ),
    ),
)
@pytest.mark.parametrize("owner", (0, 1))
def test_full_width_comparator_rejects_one_bit_drift_from_either_owner(
    tmp_path: Path,
    field: str,
    mismatch_name: str,
    owner: int,
) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture, full_width_boundaries=True)
    with np.load(capture, allow_pickle=False) as handle:
        values = {name: np.asarray(handle[name]).copy() for name in handle.files}
    target = values[field][owner].reshape(-1)
    if target.dtype == np.uint16:
        target[0] ^= np.uint16(1)
    else:
        target.view(np.uint32)[0] ^= np.uint32(1)
    np.savez(capture, **values)
    report = _compare_full_width(capture)
    assert report["exact"] is False
    assert report["status"] == "NUMERICAL_REJECTED"
    assert report["mismatch_counts"][mismatch_name] == 1


def test_feature2_rejection_recovery_is_cpu_only_distinct_and_default_off() -> None:
    recovery = (
        Path(__file__).parents[3]
        / "scripts/greenfield/recover_pp16_feature2_numerical_rejection.sh"
    )
    text = recovery.read_text()
    environment = {
        "GLM_GREENFIELD_PP16_FEATURE2_RECOVER": "0",
        "GLM_GREENFIELD_PP16_FEATURE2_RECOVERY_MODE": "off",
    }
    completed = subprocess.run(
        ["bash", str(recovery)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "RUN_PIN=363a52b7c8a4201ffbbb899352b7159c26a95b80" in text
    assert (
        "ORIGINAL_LEDGER_SHA="
        "d857d2e98cf18eb505d447acd52acf062106af75e83ac0918fa64c13105e96c2"
    ) in text
    assert (
        "CAPTURE_SHA=be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
    ) in text
    assert (
        "POST_CENSUS_SHA="
        "cd1f25488e876acf475068131c5eb7f1ca25dabea158c553de86483aa8a40798"
    ) in text
    assert "RECOVERY_REMOTE=$APPROVED_BUCKET/results/$RECOVERY_TAG" in text
    assert "RECOVERY_MODE == validate_only" in text
    assert text.index("if [[ $RECOVERY_MODE == validate_only ]]") < text.index(
        'upload_preterminal "$RECOVERY_DIR/recovery.evidence.sha256"'
    )
    assert "VALIDATED_ONLY" in text
    assert '"$SOURCE_REMOTE/NUMERICAL_REJECTED"' in text
    assert '"$SOURCE_REMOTE/SUCCESS"' in text
    assert '"in_process_release_passed": False' in text
    assert '"terminal_cleanup_authentication_passed": True' in text
    assert "72_812_032" in text
    assert '"event1_positions": 1852' in text
    assert '"carried_bfloat16_bits": 968' in text
    assert ("3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2") in text
    assert ("35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c") in text
    assert "--if-generation-match=0" in text
    assert 'gcloud storage rm --if-generation-match="$generation"' in text
    assert "rollback_unverified_terminal" in text
    assert "terminal_verified=1\ntrap - EXIT" in text
    assert '"cat", f"{uri}#{generation}"' in text
    assert 'metadata.get("crc32c_hash", metadata.get("crc32c"))' in text
    assert "[[ -s $receipt ]] || return 1" in text
    assert "assert matches == [(remote, generation)]" in text
    assert 'sync -f "$RECOVERY_DIR/upload_receipts/terminal_create.stderr"' in text
    assert '"gate_d_passed": False' in text
    assert '"performance_claim": False' in text
    assert "JAX_PLATFORMS=cpu" in text
    assert "gcloud compute" not in text
    assert "TPU_VISIBLE_DEVICES" not in text
    assert "gcloud storage cp --no-clobber" not in text


@pytest.mark.skipif(
    not SOURCES_AVAILABLE or not PROTECTED_CAPTURE.is_file(),
    reason="protected feature2 numerical sources unavailable",
)
def test_protected_feature2_rejection_exact_identities_are_executable() -> None:
    report = _compare(PROTECTED_CAPTURE)
    sealed_bytes = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    assert sha256(sealed_bytes).hexdigest() == (
        "7882af406cadfaa27191297ed44a74028f4227c52bb7f15cf17b0f9624981b45"
    )
    assert report["status"] == "NUMERICAL_REJECTED"
    assert report["exact"] is False
    assert report["capture_sha256"] == (
        "be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
    )
    assert report["mismatch_counts"] == {
        "carried_bfloat16_bits": 968,
        "contract_valid": 0,
        "event1_positions": 1852,
        "event1_scores": 2048,
        "event1_valid_counts": 0,
        "layer1_current_key_bfloat16_bits": 0,
    }
    assert (
        report["captured_array_sha256"]["current_carried_halves_bfloat16_bits"]
        == "3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2"
    )
    assert report["expected_sha256"]["carried_bfloat16_bits"] == (
        "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
    )


@pytest.mark.skipif(not DB550.is_file(), reason="protected DB550 leaves unavailable")
def test_real_db550_leaves_are_exact_through_feature_half_reducer() -> None:
    program = r"""
from hashlib import sha256
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
    replay_pp16_strategy_nd_y_x_z_bits,
)
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
    _reduce_strategy_nd_feature_half_bf16_partials,
)

path = Path("/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz")
with np.load(path, allow_pickle=False) as handle:
    bits = np.ascontiguousarray(handle["dense_virtual_partials_bfloat16_bits"])
    residual = np.ascontiguousarray(handle["post_attention_residual_bfloat16_bits"])
source = bits.reshape(2, 16, 1, 6144).view(ml_dtypes.bfloat16)
mesh = Mesh(np.asarray(jax.devices()), ("stage",))
half0 = jax.device_put(source[..., :3072], NamedSharding(mesh, P("stage", None, None, None)))
half1 = jax.device_put(source[..., 3072:], NamedSharding(mesh, P("stage", None, None, None)))

def mapped(local0, local1):
    reduced = _reduce_strategy_nd_feature_half_bf16_partials(
        local0[0], local1[0], axis_name="stage", pairs=((0, 1), (1, 0))
    )
    return reduced[None, ...]

execute = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(P("stage", None, None, None), P("stage", None, None, None)),
    out_specs=P("stage", None, None),
    check_vma=False,
)
lowered = jax.jit(execute).lower(half0, half1)
hlo = lowered.as_text()
halves = np.asarray(jax.jit(execute)(half0, half1))
actual = np.ascontiguousarray(np.concatenate((halves[0], halves[1]), axis=-1)).view(np.uint16)
model_ids = tuple(int(item) for item in np.argsort(np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)))
expected, expected_carried = derive_expected_dense_boundary_bits(bits, residual, model_ids)
full = replay_pp16_strategy_nd_y_x_z_bits(bits)
actual_carried = np.asarray(
    actual.view(ml_dtypes.bfloat16).astype(np.float32)
    + residual.view(ml_dtypes.bfloat16).astype(np.float32),
    dtype=ml_dtypes.bfloat16,
).view(np.uint16)
print(json.dumps({
    "actual_carried_sha256": sha256(actual_carried.tobytes()).hexdigest(),
    "actual_dense_sha256": sha256(actual.tobytes()).hexdigest(),
    "all_gather_count": hlo.count("stablehlo.all_gather"),
    "all_reduce_count": hlo.count("stablehlo.all_reduce"),
    "carried_mismatch_count": int(np.count_nonzero(actual_carried != expected_carried)),
    "collective_permute_count": hlo.count("stablehlo.collective_permute"),
    "dense_mismatch_count": int(np.count_nonzero(actual != expected)),
    "expected_carried_sha256": sha256(expected_carried.tobytes()).hexdigest(),
    "expected_dense_sha256": sha256(expected.tobytes()).hexdigest(),
    "forbidden_full_hidden": "tensor<1x6144xbf16>" in hlo,
    "full_replay_sha256": sha256(full.tobytes()).hexdigest(),
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "device_count": jax.device_count(),
    "payload_present": "tensor<4x1x3072xbf16>" in hlo,
    "source_file_sha256": sha256(path.read_bytes()).hexdigest(),
    "stablehlo_byte_count": len(hlo.encode()),
    "stablehlo_sha256": sha256(hlo.encode()).hexdigest(),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    report["replay_program_sha256"] = sha256(program.encode()).hexdigest()
    assert report == {
        "actual_carried_sha256": "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c",
        "actual_dense_sha256": "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc",
        "all_gather_count": 0,
        "all_reduce_count": 0,
        "backend": "cpu",
        "carried_mismatch_count": 0,
        "collective_permute_count": 1,
        "dense_mismatch_count": 0,
        "device_count": 2,
        "expected_carried_sha256": "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c",
        "expected_dense_sha256": "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc",
        "forbidden_full_hidden": False,
        "full_replay_sha256": "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc",
        "jax": "0.10.1",
        "payload_present": True,
        "replay_program_sha256": "3910d1ea062222df05deabc06b8f0ce711d8d59f948330c82e2d3da6f4cda1aa",
        "source_file_sha256": "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298",
        "stablehlo_byte_count": 23203,
        "stablehlo_sha256": "5a8b9e36d685eee5c938d62c54a826291466156321540e81d99d5113bf5d8b55",
    }
