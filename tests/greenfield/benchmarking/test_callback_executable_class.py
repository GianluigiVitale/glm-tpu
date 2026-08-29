from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.callback_executable_class import (
    CLASSIFICATION,
    TOKEN_BUCKETS,
    RunSpec,
    _inventory,
    extract_run_fingerprints,
    validate_accepted_oracle,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


def _log(
    *,
    pid: int = 17,
    glog_date: str = "I0829",
    app_date: str = "08-29",
    code_pin: str = "0123456789ab",
    marker: str = "unit-run-tag",
    buckets: tuple[int, ...] = TOKEN_BUCKETS,
    triples: tuple[tuple[str, str, str], ...] | None = None,
) -> bytes:
    triples = triples or tuple(
        (
            f"{index + 1:064x}",
            f"{index + 51:064x}",
            f"{index + 101:064x}",
        )
        for index in range(len(TOKEN_BUCKETS))
    )
    lines = [
        f"(EngineCore pid={pid}) INFO {app_date} GLM_CODE_FINGERPRINT: git={code_pin} dirty=0",
        marker,
        f"(EngineCore pid={pid}) INFO {app_date} Initializing a V1 LLM engine with compilation_config={{'debug_dump_path': None}}",
    ]
    for tokens, (executable, including_data, host) in zip(
        buckets, triples, strict=True
    ):
        lines.extend(
            (
                f"(EngineCore pid={pid}) {glog_date} (HLO module jit_step_fun_impl): Executable fingerprint:{executable}",
                f"(EngineCore pid={pid}) {glog_date} (HLO module jit_step_fun_impl): Executable fingerprint (including data segments):{including_data}",
                f"(EngineCore pid={pid}) {glog_date} (HLO module jit_step_fun_impl): Host transfer fingerprint:{host}",
                f"(EngineCore pid={pid}) INFO {app_date} Compilation of worker0 backbone --> {{'num_tokens': {tokens}, 'num_reqs': 1}} finished",
            )
        )
    return ("\n".join(lines) + "\n").encode()


def _spec(
    raw: bytes, *, triples: tuple[tuple[str, str, str], ...] | None = None
) -> RunSpec:
    triples = triples or tuple(
        (
            f"{index + 1:064x}",
            f"{index + 51:064x}",
            f"{index + 101:064x}",
        )
        for index in range(len(TOKEN_BUCKETS))
    )
    return RunSpec(
        "unit",
        sha256(raw).hexdigest(),
        17,
        "I0829",
        "08-29",
        "0123456789ab",
        "unit-run-tag",
        triples,
    )


def test_extract_run_fingerprints_binds_anchors_order_and_pairs() -> None:
    raw = _log()
    report = extract_run_fingerprints(raw, _spec(raw))
    assert [item["num_tokens"] for item in report["fingerprints"]] == list(
        TOKEN_BUCKETS
    )
    assert report["engine_pid"] == 17


def test_extract_run_fingerprints_rejects_tampered_log() -> None:
    raw = _log()
    with pytest.raises(BenchmarkValidationError, match="log SHA drifted"):
        extract_run_fingerprints(raw + b"tampered\n", _spec(raw))


def test_extract_run_fingerprints_rejects_reordered_bucket() -> None:
    raw = _log(buckets=(64, 32, 128, 256, 512, 1024, 2048))
    with pytest.raises(BenchmarkValidationError, match="token-bucket order drifted"):
        extract_run_fingerprints(raw, _spec(raw))


def test_extract_run_fingerprints_rejects_dropped_or_duplicate_triple() -> None:
    original_triples = tuple(
        (
            f"{index + 1:064x}",
            f"{index + 51:064x}",
            f"{index + 101:064x}",
        )
        for index in range(len(TOKEN_BUCKETS))
    )
    duplicate_triples = (
        original_triples[:3] + (original_triples[2],) + original_triples[4:]
    )
    raw = _log(triples=duplicate_triples)
    with pytest.raises(
        (BenchmarkValidationError, ValueError), match="fingerprint order drifted|zip"
    ):
        extract_run_fingerprints(raw, _spec(raw, triples=original_triples))


def test_extract_run_fingerprints_rejects_reordered_triple_fields() -> None:
    raw = _log()
    lines = raw.decode().splitlines()
    lines[3], lines[4] = lines[4], lines[3]
    reordered = ("\n".join(lines) + "\n").encode()
    with pytest.raises(
        BenchmarkValidationError, match="fingerprint triple order drifted"
    ):
        extract_run_fingerprints(reordered, _spec(reordered))


def test_extract_run_fingerprints_rejects_date_only_decoy() -> None:
    raw = _log(pid=99)
    spec = replace(_spec(raw), engine_pid=17)
    with pytest.raises(BenchmarkValidationError, match="run anchors are missing"):
        extract_run_fingerprints(raw, spec)


def test_extract_run_fingerprints_rejects_non_none_debug_dump() -> None:
    raw = _log().replace(b"'debug_dump_path': None", b"'debug_dump_path': '/tmp/hlo'")
    with pytest.raises(
        BenchmarkValidationError, match="debug-dump configuration drifted"
    ):
        extract_run_fingerprints(raw, _spec(raw))


def test_extract_run_fingerprints_rejects_any_wrong_anchored_code_marker() -> None:
    raw = _log()
    decoy = (
        b"(EngineCore pid=17) INFO 08-29 GLM_CODE_FINGERPRINT: "
        b"git=ffffffffffff dirty=0\n"
    )
    hostile = raw + decoy
    with pytest.raises(BenchmarkValidationError, match="code fingerprint drifted"):
        extract_run_fingerprints(hostile, _spec(hostile))


def test_classification_name_is_explicitly_non_gate() -> None:
    assert CLASSIFICATION == (
        "CALLBACK_EXECUTABLE_CLASS_REJECTED;"
        "ACCEPTED_CLASS_HAS_NO_SEALED_HLO_RECOVERY_PATH"
    )


def test_inventory_rejects_compiler_artifact_named_object(tmp_path: Path) -> None:
    forbidden = tmp_path / "sealed.optimized-hlo.txt"
    forbidden.write_text("not actually HLO")
    listing = "sealed.optimized-hlo.txt\n"
    with pytest.raises(
        BenchmarkValidationError, match="compiler-artifact-named object"
    ):
        _inventory(
            tmp_path,
            prefix="",
            expected_count=1,
            expected_sha=sha256(listing.encode()).hexdigest(),
        )


def test_accepted_oracle_seal_refuses_byte_drift() -> None:
    accepted = Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
        "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z"
    )
    if not accepted.is_dir():
        pytest.skip("protected accepted oracle unavailable")
    manifest = (accepted / "oracle/manifest.json").read_bytes()
    success = (accepted / "SUCCESS").read_bytes()
    report = validate_accepted_oracle(manifest, success)
    assert report["run_id"] == 485
    assert report["item_row_id"] == 1769
    with pytest.raises(BenchmarkValidationError, match="manifest bytes drifted"):
        validate_accepted_oracle(manifest + b"\n", success)
    with pytest.raises(BenchmarkValidationError, match="SUCCESS bytes drifted"):
        validate_accepted_oracle(manifest, success + b"\n")


def test_real_source_certificate_matches_tracked_artifact() -> None:
    repo = Path(__file__).resolve().parents[3]
    accepted = Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
        "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z"
    )
    historical = Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/dense_boundary/8k/"
        "greenfield_legacy_layer0_dense_boundary_p8155_20260814T134610676758377Z"
    )
    current = Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/layer1_rms_input/8k/"
        "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z"
    )
    if not all(path.is_dir() for path in (accepted, historical, current)):
        pytest.skip("protected callback source archives unavailable")
    from glm_tpu.greenfield.benchmarking.callback_executable_class import (
        classify_callback_executable_class,
        serialize_callback_executable_class,
    )

    payload = serialize_callback_executable_class(
        classify_callback_executable_class(
            accepted,
            historical_root=historical,
            current_root=current,
            rejection_artifact_path=(
                repo
                / "docs/artifacts/layer1-rms-input-observer-perturbation-rejection.json"
            ),
        )
    )
    assert (
        payload
        == (
            repo / "docs/artifacts/callback-executable-class-certificate.json"
        ).read_bytes()
    )
