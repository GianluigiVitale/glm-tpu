from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.greenfield.prepare_compact_failure_archive as compact_archive
from glm_tpu.greenfield.validation.layer1_rms_input import (
    REJECTED_RUN_TAG,
    REJECTION_CLASSIFICATION,
    REJECTION_MESSAGE,
    Layer1RmsInputCaptureConfig,
    capture_accepted_layer1_rms_input,
    validate_layer1_rms_input_artifacts,
)
from scripts.greenfield.build_rejected_layer1_prune_manifest import (
    BUCKET,
    EXPECTED,
    PAYLOAD_PREFIX,
    build_manifest,
)
from scripts.greenfield.prepare_compact_failure_archive import (
    prepare_compact_failure_archive,
)

REPO = Path(__file__).resolve().parents[3]


def _config(root: Path) -> Layer1RmsInputCaptureConfig:
    return Layer1RmsInputCaptureConfig(
        source_dump_dir=root / "missing-source",
        output_dir=root / "must-not-exist",
        db550_boundary_path=root / "missing-db550",
        straddler_classification_path=root / "missing-straddler",
        vllm_repository=root / "missing-vllm",
        expected_run_tag="stale-caller",
        expected_legacy_code_hash="0" * 40,
        expected_oracle_pin="1" * 40,
    )


@pytest.mark.parametrize(
    "operation",
    [capture_accepted_layer1_rms_input, validate_layer1_rms_input_artifacts],
)
def test_sealer_and_validator_tombstone_before_io(tmp_path: Path, operation) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match=REJECTION_CLASSIFICATION):
        operation(config)
    assert not config.output_dir.exists()
    assert REJECTED_RUN_TAG in REJECTION_MESSAGE
    assert "557434" in REJECTION_MESSAGE
    assert "573438" in REJECTION_MESSAGE


def _zero_external_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "external-command-was-called"
    (fake_bin / "bash").symlink_to("/bin/bash")
    for command in ("date", "gcloud", "git", "python", "python3"):
        executable = fake_bin / command
        executable.write_text(
            '#!/bin/sh\nprintf \'%s\\n\' "$0 $*" >>"$EXTERNAL_SENTINEL"\nexit 99\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "EXTERNAL_SENTINEL": str(sentinel),
            "GLM_GREENFIELD_LAYER1_RMS_INPUT_CAPTURE_TAG": "fixed_test_tag",
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        }
    )
    return environment, sentinel


@pytest.mark.parametrize(
    ("script", "extra_environment"),
    [
        ("scripts/greenfield/run_capture_legacy_layer1_rms_input.sh", {}),
        (
            "scripts/greenfield/run_capture_short_context_dsa_oracle.sh",
            {
                "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE": "1",
                "GLM_GREENFIELD_DSA_INTERNALS_MODE": "layer1_rms_input",
            },
        ),
    ],
)
def test_shell_entrypoints_refuse_before_external_action(
    tmp_path: Path, script: str, extra_environment: dict[str, str]
) -> None:
    environment, sentinel = _zero_external_environment(tmp_path)
    environment.update(extra_environment)
    completed = subprocess.run(
        ["/bin/bash", str(REPO / script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert REJECTION_CLASSIFICATION in completed.stderr
    assert not sentinel.exists()


def test_python_cli_refuses_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/greenfield/capture_accepted_layer1_rms_input.py"),
            "--source-dump-dir",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--db550-boundary",
            str(tmp_path / "db550"),
            "--straddler-classification",
            str(tmp_path / "straddler"),
            "--vllm-repository",
            str(tmp_path / "vllm"),
            "--run-tag",
            "stale",
            "--legacy-code-hash",
            "0" * 40,
            "--oracle-pin",
            "1" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert REJECTION_CLASSIFICATION in completed.stderr
    assert not output.exists()


def test_compact_failure_archive_excludes_reproducible_bulk(tmp_path: Path) -> None:
    source = tmp_path / REJECTED_RUN_TAG
    (source / "source_dumps/w4").mkdir(parents=True)
    (source / "source_dumps/w4/topk.step0004.evt00.proc0.npz").write_bytes(
        b"topk" * 100
    )
    (source / "source_dumps/w4/internals.layer1.proc0.npz").write_bytes(
        b"irreplaceable-internal"
    )
    (source / "vllm_deadbeef.tar.gz").write_bytes(b"reproducible-vllm")
    (source / "observer_deadbeef.bundle").write_bytes(b"reproducible-observer")
    (source / "orchestrator.log").write_text("failure\n", encoding="utf-8")
    output = tmp_path / "compact"

    report = prepare_compact_failure_archive(source, output)

    assert report["excluded_file_count"] == 3
    assert report["kept_file_count"] == 2
    assert (output / "source_dumps/w4/internals.layer1.proc0.npz").is_file()
    assert (output / "orchestrator.log").is_file()
    assert (output / "orchestrator.log").stat().st_ino != (
        source / "orchestrator.log"
    ).stat().st_ino
    assert not (output / "source_dumps/w4/topk.step0004.evt00.proc0.npz").exists()
    manifest = json.loads((output / "failure_archive_manifest.json").read_text())
    assert manifest["manifest_sha256"] == report["manifest_sha256"]
    reasons = {item["reason"] for item in manifest["excluded"]}
    assert reasons == {
        "reproducible_dsa_source_dump",
        "reproducible_observer_git_bundle",
        "reproducible_vllm_source_archive",
    }

    wrong_run = tmp_path / "unrelated-run"
    wrong_run.mkdir()
    with pytest.raises(ValueError, match="bound to the rejected layer-1 run"):
        prepare_compact_failure_archive(wrong_run, tmp_path / "wrong-compact")


def test_compact_failure_archive_refuses_path_aliases_and_reserved_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / REJECTED_RUN_TAG
    source.mkdir()
    (source / "diagnostic.json").write_text("{}\n", encoding="utf-8")
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="source root must not be a symlink"):
        prepare_compact_failure_archive(source_link, tmp_path / "compact-source-link")

    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(ValueError, match="output root must not be a symlink"):
        prepare_compact_failure_archive(source, output_link)

    with pytest.raises(ValueError, match="source and output must be disjoint"):
        prepare_compact_failure_archive(source, source / "compact")

    (source / "failure_archive_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved failure archive manifest"):
        prepare_compact_failure_archive(source, tmp_path / "compact-reserved")


def test_compact_failure_archive_refuses_excluded_source_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / REJECTED_RUN_TAG
    (source / "source_dumps/w4").mkdir(parents=True)
    excluded = source / "source_dumps/w4/topk.step0004.evt00.proc0.npz"
    excluded.write_bytes(b"original")
    (source / "diagnostic.json").write_text("{}\n", encoding="utf-8")
    original_sha256 = compact_archive._sha256
    mutated = False

    def mutate_after_first_hash(path: Path) -> str:
        nonlocal mutated
        digest = original_sha256(path)
        if path == excluded and not mutated:
            excluded.write_bytes(b"mutated-after-snapshot")
            mutated = True
        return digest

    monkeypatch.setattr(compact_archive, "_sha256", mutate_after_first_hash)
    with pytest.raises(RuntimeError, match="source entry drifted"):
        prepare_compact_failure_archive(source, tmp_path / "compact-mutation")


def test_shared_failure_trap_preserves_full_evidence_for_active_modes() -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text(encoding="utf-8")
    assert "prepare_compact_failure_archive.py" not in wrapper
    assert '"$REMOTE_PREFIX/diagnostic_local/"' in wrapper


def test_prune_manifest_requires_exact_semantic_identity_and_inventory() -> None:
    def item(name: str, size: int, generation: int) -> dict[str, object]:
        return {
            "metadata": {
                "bucket": BUCKET,
                "crc32c": "AAAAAA==",
                "generation": str(generation),
                "name": PAYLOAD_PREFIX + name,
                "size": str(size),
            }
        }

    topk_count = EXPECTED["delete"]["file_count"] - 2
    topk_total = EXPECTED["delete"]["byte_count"] - 11 - 13
    base, remainder = divmod(topk_total, topk_count)
    listing = [
        item(
            f"source_dumps/w4/topk.step{index:04d}.evt00.proc0.npz",
            base + (index < remainder),
            1000 + index,
        )
        for index in range(topk_count)
    ]
    listing.extend(
        [
            item("vllm_deadbeef.tar.gz", 11, 2000),
            item("observer_deadbeef.bundle", 13, 2001),
            *[
                item(
                    f"compact/{index}.json",
                    EXPECTED["keep"]["byte_count"] // EXPECTED["keep"]["file_count"]
                    + (
                        index
                        < EXPECTED["keep"]["byte_count"]
                        % EXPECTED["keep"]["file_count"]
                    ),
                    3000 + index,
                )
                for index in range(EXPECTED["keep"]["file_count"])
            ],
        ]
    )
    with pytest.raises(ValueError, match="semantic identity drifted"):
        build_manifest(listing)
    listing.pop()
    with pytest.raises(ValueError, match="inventory drifted"):
        build_manifest(listing)

    duplicate = [item("duplicate.json", 1, 1), item("duplicate.json", 1, 2)]
    with pytest.raises(ValueError, match="duplicate failure-object name"):
        build_manifest(duplicate)

    invalid_generation = item("invalid.json", 1, 1)
    invalid_generation["metadata"]["generation"] = "not-numeric"
    with pytest.raises(ValueError, match="invalid failure-object generation"):
        build_manifest([invalid_generation])
