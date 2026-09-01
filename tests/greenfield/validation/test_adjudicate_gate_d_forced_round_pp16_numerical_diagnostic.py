from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]


def _load(relative: str, name: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "scripts/greenfield/"
    "adjudicate_gate_d_forced_round_pp16_numerical_diagnostic.py",
    "_gate_d_forced_round_diagnostic_adjudicator",
)
PUBLISHER = _load(
    "scripts/greenfield/publish_gate_d_forced_round_pp16_numerical.py",
    "_gate_d_forced_round_publisher_current",
)
DRIVER = MODULE._load_committed_module(
    MODULE.DRIVER_PATH, MODULE.RUN_PIN, "_gate_d_forced_round_driver_historical"
)


def _load_for_local(relative: str, pin: str, name: str) -> ModuleType:
    del pin, name
    if relative == MODULE.PUBLISHER_PATH:
        return PUBLISHER
    if relative == MODULE.DRIVER_PATH:
        return DRIVER
    raise AssertionError(relative)


def _local_evidence(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(MODULE, "_load_committed_module", _load_for_local)
    monkeypatch.setattr(
        MODULE,
        "_validate_remote",
        lambda records, terminal, snapshots, ledger_raw: [
            {
                "crc32c": terminal["crc32c"],
                "generation": terminal["generation"],
                "path": "diagnostic_objects.json",
                "sha256": terminal["sha256"],
                "size": terminal["size"],
            }
        ],
    )
    return MODULE.adjudicate()


def test_local_diagnostic_is_independently_reclassified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _local_evidence(monkeypatch)
    assert report["classification"] == MODULE.FINAL_CLASSIFICATION
    assert report["forced_round_mechanism_accepted"] is False
    assert report["first_observed_divergence"] == "layer1.query projection output"
    assert report["tpu_numerical_execution_performed"] is True
    assert report["tpu_rerun_performed"] is False
    numerical = report["independent_numerical_classification"]
    assert numerical["candidate_watchpoint_matches"][
        "layer1.normalized:owner0"
    ]
    assert not numerical["candidate_watchpoint_matches"]["layer1.query:owner0"]


def test_ledger_bytes_are_fixed(tmp_path: Path) -> None:
    run = tmp_path / MODULE.RUN_TAG
    shutil.copytree(MODULE.RUN_DIR, run)
    ledger = run / "diagnostic_objects.json"
    ledger.chmod(0o600)
    value = json.loads(ledger.read_bytes())
    value["objects"][0]["size"] += 1
    ledger.write_bytes(MODULE._canonical_json(value))
    with pytest.raises(MODULE.DiagnosticValidationError, match="ledger bytes"):
        MODULE._validate_ledger(run)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "hash", "extra"))
def test_dependency_exception_is_exactly_one_committed_helper(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    runner = json.loads((MODULE.RUN_DIR / "runner.json").read_bytes())
    dependencies_raw = (MODULE.RUN_DIR / "dependencies.json").read_bytes()
    dependencies = json.loads(dependencies_raw)
    records = dependencies["python_modules"]
    helper = next(
        record
        for record in records
        if record["path"] == str(MODULE.FORCED_HLO_HELPER_PATH)
    )
    if mutation == "missing":
        records.remove(helper)
    elif mutation == "duplicate":
        records.append(deepcopy(helper))
    elif mutation == "hash":
        helper["sha256"] = "0" * 64
    else:
        helper["unbound"] = False
    primitives = PUBLISHER._load_primitives(MODULE.RUN_PIN)
    with pytest.raises((MODULE.DiagnosticValidationError, RuntimeError)):
        MODULE._validate_dependencies_with_exact_helper(
            PUBLISHER, primitives, runner, dependencies, dependencies_raw
        )


@pytest.mark.parametrize("pin_name", ("run", "acquisition"))
def test_dependency_exception_rejects_either_git_pin_drifting(
    monkeypatch: pytest.MonkeyPatch, pin_name: str
) -> None:
    runner = json.loads((MODULE.RUN_DIR / "runner.json").read_bytes())
    dependencies_raw = (MODULE.RUN_DIR / "dependencies.json").read_bytes()
    dependencies = json.loads(dependencies_raw)
    primitives = PUBLISHER._load_primitives(MODULE.RUN_PIN)
    target = (
        MODULE.RUN_PIN if pin_name == "run" else MODULE.HLO_ACQUISITION_PIN
    )
    original = MODULE._git

    def git(*arguments: str) -> bytes:
        if arguments == (
            "show",
            f"{target}:{MODULE.FORCED_HLO_HELPER_SOURCE_PATH}",
        ):
            return b"drift"
        return original(*arguments)

    monkeypatch.setattr(MODULE, "_git", git)
    with pytest.raises(MODULE.DiagnosticValidationError, match="helper bytes"):
        MODULE._validate_dependencies_with_exact_helper(
            PUBLISHER, primitives, runner, dependencies, dependencies_raw
        )


def _remote_fixture() -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, bytes],
    bytes,
    list[dict[str, object]],
]:
    records, ledger_raw = MODULE._validate_ledger(MODULE.RUN_DIR)
    snapshots = MODULE._validate_local_members(MODULE.RUN_DIR, records)
    terminal = MODULE._validate_receipt(MODULE.RUN_DIR, ledger_raw)
    expected = {**records, "diagnostic_objects.json": terminal}
    listing = []
    prefix = MODULE.REMOTE.rstrip("/") + "/"
    for path, record in sorted(expected.items()):
        listing.append(
            {
                "metadata": {
                    "crc32c": record["crc32c"],
                    "generation": record["generation"],
                    "size": str(record["size"]),
                },
                "type": "cloud_object",
                "url": f"{prefix}{path}#{record['generation']}",
            }
        )
    return records, terminal, snapshots, ledger_raw, listing


def _remote_bytes(
    records: dict[str, dict[str, object]],
    terminal: dict[str, object],
    snapshots: dict[str, bytes],
    ledger_raw: bytes,
) -> dict[str, bytes]:
    result = dict(snapshots)
    result["diagnostic_objects.json"] = ledger_raw
    return result


def test_generation_qualified_remote_archive_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, terminal, snapshots, ledger_raw, listing = _remote_fixture()
    payloads = _remote_bytes(records, terminal, snapshots, ledger_raw)
    monkeypatch.setattr(
        MODULE,
        "_list_remote",
        lambda *, soft_deleted: [] if soft_deleted else listing,
    )
    monkeypatch.setattr(
        MODULE,
        "_cat_generation",
        lambda url: payloads[url.rsplit("/", 1)[-1].split("#", 1)[0]]
        if "/hlo/" not in url
        else payloads["hlo/" + url.rsplit("/", 1)[-1].split("#", 1)[0]],
    )
    manifest = MODULE._validate_remote(
        records, terminal, snapshots, ledger_raw
    )
    assert len(manifest) == 18
    assert manifest[-1]["path"] == "sync.txt"


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate",
        "soft_deleted",
        "generation",
        "crc32c",
        "bytes",
        "missing",
        "terminal_order",
    ),
)
def test_remote_archive_rejects_history_or_identity_drift(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    records, terminal, snapshots, ledger_raw, listing = _remote_fixture()
    payloads = _remote_bytes(records, terminal, snapshots, ledger_raw)
    listing = deepcopy(listing)
    soft: list[dict[str, object]] = []
    if mutation == "duplicate":
        listing.append(deepcopy(listing[0]))
    elif mutation == "soft_deleted":
        soft.append(deepcopy(listing[0]))
    elif mutation == "generation":
        listing[0]["metadata"]["generation"] = "1"
    elif mutation == "crc32c":
        listing[0]["metadata"]["crc32c"] = "AAAAAA=="
    elif mutation == "bytes":
        first = listing[0]["url"].rsplit("/", 1)[-1].split("#", 1)[0]
        payloads[first] += b"x"
    elif mutation == "terminal_order":
        terminal["generation"] = "1"
        item = next(
            value
            for value in listing
            if "/diagnostic_objects.json#" in value["url"]
        )
        item["metadata"]["generation"] = "1"
        item["url"] = item["url"].split("#", 1)[0] + "#1"
    else:
        listing.pop()
    monkeypatch.setattr(
        MODULE,
        "_list_remote",
        lambda *, soft_deleted: soft if soft_deleted else listing,
    )

    def cat(url: str) -> bytes:
        name = url.rsplit("/", 1)[-1].split("#", 1)[0]
        key = "hlo/" + name if "/hlo/" in url else name
        return payloads[key]

    monkeypatch.setattr(MODULE, "_cat_generation", cat)
    with pytest.raises(MODULE.DiagnosticValidationError):
        MODULE._validate_remote(records, terminal, snapshots, ledger_raw)


def test_remote_listing_parser_rejects_prefix_escape() -> None:
    value = [{"type": "prefix", "url": "gs://wrong/"}]
    with pytest.raises(MODULE.DiagnosticValidationError, match="prefix"):
        MODULE._parse_listing(
            json.dumps(value).encode(), {MODULE.REMOTE.rstrip("/") + "/"}
        )


@pytest.mark.parametrize("soft_deleted", (False, True))
def test_remote_history_command_uses_canonical_scope(
    monkeypatch: pytest.MonkeyPatch, soft_deleted: bool
) -> None:
    observed: list[str] = []

    def run(
        command: list[str], *, environment: dict[str, str], timeout: int = 180
    ) -> subprocess.CompletedProcess[bytes]:
        del environment, timeout
        observed.extend(command)
        if soft_deleted:
            return subprocess.CompletedProcess(
                command,
                1,
                b"",
                b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n",
            )
        return subprocess.CompletedProcess(command, 0, b"[]", b"")

    monkeypatch.setattr(MODULE, "_run", run)
    assert MODULE._list_remote(soft_deleted=soft_deleted) == []
    assert ("--soft-deleted" in observed) is soft_deleted
    assert ("--all-versions" in observed) is (not soft_deleted)
    assert ("--exhaustive" in observed) is soft_deleted


def test_adjudicator_source_verifier_rejects_uncommitted_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "_git", lambda *args: b"0" * 40 + b"\n")
    with pytest.raises(MODULE.DiagnosticValidationError, match="not committed"):
        MODULE._verify_running_source("0" * 40, "0" * 64)
