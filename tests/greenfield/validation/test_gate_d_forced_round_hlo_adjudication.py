from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

import glm_tpu.greenfield.validation.gate_d_forced_round_hlo as adjudication_module
from glm_tpu.greenfield.validation.gate_d_forced_round_hlo import (
    EXPECTED_OPTIMIZED_HLO_SHA256,
    EXPECTED_REMOTE_REPLAY_PATH,
    EXPECTED_RUN_TAG,
    EXPECTED_STABLEHLO_SHA256,
    ForcedRoundHloAdjudicationError,
    _adjudicate_acquired_files,
    _read_canonical_run,
    _read_regular_at,
    adjudicate_acquired_run,
    audit_optimized_hlo_structure,
    audit_stablehlo_structure,
)

RUN = Path("/home/gianl/gate-d-runs") / EXPECTED_RUN_TAG
OPTIMIZED = RUN / "hlo/forced_round_pp16_stage0.optimized_hlo.txt"
STABLE = RUN / "hlo/forced_round_pp16_stage0.stablehlo.mlir"
ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "docs/artifacts/gate-d-forced-round-pp16-hlo-causal-adjudication.json"
)
SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/greenfield/adjudicate_gate_d_forced_round_pp16_hlo.py"
)
pytestmark = pytest.mark.skipif(
    not (OPTIMIZED.is_file() and STABLE.is_file()),
    reason="protected acquired HLO is not present",
)


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def test_exact_acquired_graph_and_archive_pass() -> None:
    report = adjudicate_acquired_run(RUN)

    assert report["classification"] == (
        "HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_LOCALITY_ACCEPTED;"
        "TPU_NUMERICAL_UNPROVEN;GATE_D_OPEN"
    )
    assert report["authorization"] == {
        "full_8k": False,
        "numerical_execution": False,
        "performance_claim": False,
        "persistence_only": True,
    }
    assert report["optimized_hlo"]["sha256"] == EXPECTED_OPTIMIZED_HLO_SHA256
    assert report["stablehlo"]["sha256"] == EXPECTED_STABLEHLO_SHA256
    assert report["remote_archive"]["terminal_last"] is True
    assert report["remote_archive"]["object_count_including_ledger_and_terminal"] == 17
    assert report["remote_archive"]["remote_replay"] == {
        "all_versions_live_generation_count": 17,
        "bucket_location": "US-CENTRAL2",
        "generation_download_count": 17,
        "sha256": "85bcd02c9b112ef3e21d065ebfae04f0d06e4a922f1d15c0f971fb669992e1d8",
        "soft_deleted_generation_count": 0,
    }


def test_committed_adjudication_is_exact_canonical_report() -> None:
    report = adjudicate_acquired_run(RUN)
    expected = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    artifact = ARTIFACT.read_bytes()

    assert artifact == expected
    assert sha256(artifact).hexdigest() == (
        "4db0ea2bfd8f34ec631213b6724d220038c08b23da788164c62e75efacbbab74"
    )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            (
                "%reduce_precision.4 = f32[1,6144]{1,0:T(1,128)} "
                "reduce-precision(%mul.173), exponent_bits=8, mantissa_bits=7"
            ),
            (
                "%reduce_precision.4 = f32[1,6144]{1,0:T(1,128)} "
                "reduce-precision(%add.198), exponent_bits=8, mantissa_bits=7"
            ),
        ),
        ("exponent_bits=8, mantissa_bits=7", "exponent_bits=8, mantissa_bits=6"),
        (
            (
                "fusion(%bitcast.135, %rsqrt.8, %param.23, %param.22), kind=kLoop, "
                "calls=%fused_computation.40"
            ),
            (
                "fusion(%bitcast.135, %rsqrt.8, %param.23, %param.24), kind=kLoop, "
                "calls=%fused_computation.40"
            ),
        ),
        (
            ("fusion(%param.23, %param.22), kind=kLoop, calls=%fused_computation.41"),
            ("fusion(%param.23, %param.24), kind=kLoop, calls=%fused_computation.41"),
        ),
        (
            (
                "%convert.21 = f32[1,6144]{1,0:T(1,128)S(3)} "
                "convert(%multiply_convert_fusion)"
            ),
            ("%convert.21 = f32[1,6144]{1,0:T(1,128)S(3)} convert(%param.22)"),
        ),
        (
            (
                "%fusion.8 = (f32[]{:T(128)}, f32[128]{0:T(128)S(3)}) "
                "fusion(%copy-done.1, %bitcast.133)"
            ),
            (
                "%fusion.8 = (f32[]{:T(128)}, f32[128]{0:T(128)S(3)}) "
                "fusion(%copy-done.1, %param.22)"
            ),
        ),
        (
            "%param.27, %multiply_convert_fusion, /*index=5*/",
            "%param.27, %param.22, /*index=5*/",
        ),
        (
            (
                "%bitcast.130 = bf16[1,1,6144]{2,1,0:T(2,128)(2,1)} "
                "bitcast(%copy-done.2)"
            ),
            ("%bitcast.130 = bf16[1,1,6144]{2,1,0:T(2,128)(2,1)} bitcast(%param.23)"),
        ),
        ("%all-gather.3 =", "%all-gather.extra ="),
        ('custom_call_target="ConcatBitcast"', 'custom_call_target="HostCallback"'),
        (
            "(bf16[1,1,6144], f32[1,1,32,128]",
            "(bf16[2,1,6144], f32[1,1,32,128]",
        ),
    ),
)
def test_optimized_hlo_hostile_edge_mutations_fail(old: str, new: str) -> None:
    hostile = _replace_once(OPTIMIZED.read_text(), old, new)

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_optimized_hlo_structure(hostile)


def test_optimized_hlo_nonlocal_collective_mutation_fails() -> None:
    source = OPTIMIZED.read_text()
    old = "replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true"
    assert source.count(old) == 3
    hostile = source.replace(
        old,
        "replica_groups={{0,1,2,3}}, dimensions={0}, use_global_device_ids=true",
        1,
    )

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_optimized_hlo_structure(hostile)


def test_qkv_loop_carried_index_mutation_fails() -> None:
    source = OPTIMIZED.read_text()
    old = (
        "%get-tuple-element.303, /*index=5*/%get-tuple-element.304, "
        "%get-tuple-element.305)"
    )
    hostile = _replace_once(
        source,
        old,
        "%get-tuple-element.302, /*index=5*/%get-tuple-element.304, "
        "%get-tuple-element.305)",
    )

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_optimized_hlo_structure(hostile)


def test_competing_unrounded_primary_consumer_fails() -> None:
    source = OPTIMIZED.read_text()
    insertion = "  %unrounded_bypass = bf16[1,6144]{1,0} copy(%param.22)\n  %copy.71 ="
    hostile = _replace_once(source, "  %copy.71 =", insertion)

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_optimized_hlo_structure(hostile)


def test_host_staging_token_fails() -> None:
    hostile = OPTIMIZED.read_text() + "\n// xla_python_cpu_callback\n"

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_optimized_hlo_structure(hostile)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("format = e8m7", "format = e8m6"),
        ("%iterArg_97 = %22", "%iterArg_97 = %7"),
        (
            "%121 = stablehlo.convert %114",
            "%121 = stablehlo.convert %7",
        ),
        (
            "sdy.return %346, %347, %348",
            "sdy.return %22, %347, %348",
        ),
    ),
)
def test_stablehlo_hostile_lineage_mutations_fail(old: str, new: str) -> None:
    hostile = _replace_once(STABLE.read_text(), old, new)

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_stablehlo_structure(hostile)


def test_stablehlo_nonlocal_collective_mutation_fails() -> None:
    source = STABLE.read_text()
    old = "replica_groups = dense<[[0, 1]]>"
    assert source.count(old) == 3
    hostile = source.replace(old, "replica_groups = dense<[[0, 1, 2, 3]]>", 1)

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_stablehlo_structure(hostile)


def test_stablehlo_host_staging_token_fails() -> None:
    hostile = STABLE.read_text() + "\n// outside_compilation\n"

    with pytest.raises(ForcedRoundHloAdjudicationError):
        audit_stablehlo_structure(hostile)


def test_archive_claim_escalation_fails() -> None:
    evidence = _read_canonical_run(RUN)
    runner = json.loads(evidence["runner.json"])
    runner["tpu_numerical_execution_performed"] = True
    evidence["runner.json"] = (json.dumps(runner, sort_keys=True) + "\n").encode()

    with pytest.raises(ForcedRoundHloAdjudicationError):
        _adjudicate_acquired_files(
            evidence,
            EXPECTED_REMOTE_REPLAY_PATH.read_bytes(),
        )


def test_terminal_generation_reordering_fails() -> None:
    evidence = _read_canonical_run(RUN)
    receipt = json.loads(evidence["terminal_upload_receipt.json"])
    receipt["terminal"]["generation"] = "1"
    evidence["terminal_upload_receipt.json"] = (
        json.dumps(receipt, sort_keys=True) + "\n"
    ).encode()

    with pytest.raises(ForcedRoundHloAdjudicationError):
        _adjudicate_acquired_files(
            evidence,
            EXPECTED_REMOTE_REPLAY_PATH.read_bytes(),
        )


def test_duplicate_remote_ledger_path_fails() -> None:
    evidence = _read_canonical_run(RUN)
    ledger = json.loads(evidence["remote_objects.json"])
    ledger["objects"][1] = {**ledger["objects"][0]}
    evidence["remote_objects.json"] = (
        json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    with pytest.raises(ForcedRoundHloAdjudicationError):
        _adjudicate_acquired_files(
            evidence,
            EXPECTED_REMOTE_REPLAY_PATH.read_bytes(),
        )


def test_remote_replay_soft_delete_claim_mutation_fails() -> None:
    replay = json.loads(EXPECTED_REMOTE_REPLAY_PATH.read_text())
    replay["all_versions_catalogue"]["soft_deleted_generation_count"] = 1
    hostile = (json.dumps(replay, sort_keys=True) + "\n").encode()

    with pytest.raises(ForcedRoundHloAdjudicationError):
        _adjudicate_acquired_files(_read_canonical_run(RUN), hostile)


def test_remote_replay_requires_exhaustive_soft_delete_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = json.loads(EXPECTED_REMOTE_REPLAY_PATH.read_text())
    replay["queries"]["soft_deleted"] = (
        "PYTHONWARNINGS=ignore gcloud storage ls --soft-deleted --json PREFIX/**"
    )
    hostile = (
        json.dumps(replay, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    monkeypatch.setattr(
        adjudication_module,
        "EXPECTED_REMOTE_REPLAY_SHA256",
        sha256(hostile).hexdigest(),
    )

    with pytest.raises(ForcedRoundHloAdjudicationError, match="claim drifted"):
        _adjudicate_acquired_files(_read_canonical_run(RUN), hostile)


def test_noncanonical_run_path_fails(tmp_path: Path) -> None:
    alias = tmp_path / EXPECTED_RUN_TAG
    alias.symlink_to(RUN, target_is_directory=True)

    with pytest.raises(ForcedRoundHloAdjudicationError, match="canonical path drifted"):
        adjudicate_acquired_run(alias)


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_same_fd_reader_rejects_links(tmp_path: Path, link_kind: str) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"sealed")
    linked = tmp_path / "linked"
    if link_kind == "symlink":
        linked.symlink_to(target)
    else:
        os.link(target, linked)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ForcedRoundHloAdjudicationError):
            _read_regular_at(directory_fd, "linked")
    finally:
        os.close(directory_fd)


def test_cli_has_stdout_only_append_only_contract(tmp_path: Path) -> None:
    forbidden = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            str(SCRIPT),
            "--output",
            str(forbidden),
        ],
        cwd=SCRIPT.parents[2],
        env={
            "HOME": "/home/gianl",
            "JAX_PLATFORMS": "cpu",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": ".",
        },
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert not forbidden.exists()
    assert "--output" not in SCRIPT.read_text()
