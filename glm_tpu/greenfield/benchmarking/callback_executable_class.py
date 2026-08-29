"""Fail-closed offline certificate for the rejected callback executable class.

This module authenticates the accepted oracle, three retained legacy logs,
and their archive inventories.  It proves fingerprint-class
equality/disjointness and the absence of compiler-artifact-named objects.  It
deliberately does not infer which XLA operation changed or substitute CPU IR
for a missing TPU executable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..errors import BenchmarkValidationError

TOKEN_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
CLASSIFICATION = (
    "CALLBACK_EXECUTABLE_CLASS_REJECTED;ACCEPTED_CLASS_HAS_NO_SEALED_HLO_RECOVERY_PATH"
)
TOMBSTONE_PIN = "c7973435aa2fc948da9185ef99938f886613ce2f"
REJECTION_ARTIFACT_SHA256 = (
    "d164972602b7d69c478b7afe0d63f6e9df2ff9d48d354f67565b9369ce4a5b1a"
)
ACCEPTED_MANIFEST_FILE_SHA256 = (
    "62c3fc2ad45d368c91cc41901947a69abf7a8c7abb57fb56b088871e90e5f1d3"
)
ACCEPTED_MANIFEST_SHA256 = (
    "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
)
ACCEPTED_SUCCESS_SHA256 = (
    "0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9"
)
ACCEPTED_FULL_CODE_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
ACCEPTED_REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z"
)
ACCEPTED_SUCCESS_FIELDS = {
    "artifact_kind": "greenfield_short_context_legacy_dsa_oracle",
    "sealer_code_hash": "89a4363caaf93edd8abbf7fc6608f6edddb825df",
    "source_capture_code_hash": "b5adc6385fda39cfaa7deed5bcef5d06edaaa511",
    "manifest_sha256": ACCEPTED_MANIFEST_SHA256,
    "source_run_id": "485",
    "source_item_row_id": "1769",
    "source_dump_file_count": "294",
    "evidence_sha256": "d84ca2f931c44cc413f017928162b9292a50acdf3c0f92b1a0b2efee6a3d4a28",
    "remote_objects_sha256": "26848233b8d1c1beea1a165a2a9c1db0baaf0f3c5536c72897a050a49a5da074",
    "remote_prefix": ACCEPTED_REMOTE_PREFIX,
}

ACCEPTED_FINGERPRINTS = (
    (
        "ae38e199c39829fdeba153a962b45002c399742fb7fe42176978faa91389ba03",
        "817c14978c9a8e14a7ebae424f0b1adc140cd531ba3e0d5fae728a719f8be3fb",
        "36bb8ce2dd4a253354efc7e6b820bbc06cdb0cf86cddc7e5a43657bb1f2d1b06",
    ),
    (
        "a4c3a9f7efbe0efd1f174b6d930268e365c4bac26a6b09d3f0873321bbe296d1",
        "d52c3aac463161f1ddcae08b5c86701f42fe86d2843ebf90a03595f82e64d7a0",
        "9698b4afc8abd85d5fab721c96cb1d8562e34834bd828282fab6f8b73b7fe687",
    ),
    (
        "aae23675b53a7e352e1a5dc6c253ccdc177db2e303502f1bce4130c32999dbe4",
        "e5940c4075fc7bf6233a321e175615f04e89014247ed01e3368477605f3d17d8",
        "90a5e7be983b82073bfeba82037b1ff516a06a85544d0fa5afcc44facd66c06e",
    ),
    (
        "e4cb00e0d2d4fea078cc26c51e5f67734db709e42aa7f4359f94bf4eed249ee2",
        "7e09135faab89dfbcb54024e46b01af1b06f567076aae9f2ce0895c84fba4968",
        "b723e7e76bc624e47a0d2e2c9fe432c1535a3b65ece504fc954e37f102da13ae",
    ),
    (
        "5cd42be77bc7e8fa62b9c69a4b633e1f62d8f38c4c9483d5685268e214eceb53",
        "b16797bbca1589f78eb7c42385657ab1fb3ac788158761a7ade21b963063ce5a",
        "f91818d8c0cc0f499feb95270c56618176a5daedb1259a77fdefcb002a5ccf4f",
    ),
    (
        "5645e5bae10fe9f7302b3daa1ef7cdda6ae9b889bde805bb19b7e45aa923c4f5",
        "ff17861ea4208c73f8cefe57d32626651c8f814c6639d9d1ea230f7948dcd4d3",
        "1015c7ae0aaa998dd6e6837a70fb501d03054b31a53d96b550b428c6c456e61e",
    ),
    (
        "2d55b18584af02fc601e5da6481fc1c965da052aa962261e99047cf15088f7d3",
        "20cbe1c4b34c03339f7677062fc594f44e1ca99b5131a409f3935d071fdb067b",
        "814f3138915ea396770a6eb969eb944b4a913eafbf2a3e863691142d48fc3e71",
    ),
)
CALLBACK_FINGERPRINTS = (
    (
        "ac610a5f0b98fc53d0902434c9f6237a520c87c7e6ccef243b00fca2f9788d07",
        "f72fd4e1877fd1644ba98666778a35307b5466220272c39adaa0168392977c2d",
        "4afbae49cb3948a36d5922fc8e75835a1be6d2544bb513be64ddeed4e6212829",
    ),
    (
        "a9fa93a83b06894a12d60b624cbba2f0f38e5c1c37705fd97af4e250147095e0",
        "ec93ef38f6a6ae91eddd96103dfa547056b0692a743a144c6e84c01dad2f1223",
        "d63222b2ddc38d881074fc1b416b55d669a0fad22298e3f50c4656f62167002c",
    ),
    (
        "dc3b52c1704780f65c455441bdbbb0faa57a8345976fdff137b58dd54732e857",
        "1eb90565e79f051cd5fc5ad42338bab45efd7a8674868bae9b4485b6e88901e3",
        "16d0a5b7a07797c6f968d8392c5eeb52972483e8cf0d886b91f57d4c1d2d6b5b",
    ),
    (
        "754ea08401eabe2eeb2131b9f9b3b3551453b3a50e006141e11b30c189981114",
        "f05955ab4fbd08a6cd9ec76b00ba182c34f63e46bcae40c0bbe84223c8238ce1",
        "5103974d54d35435da13d6edbf6c0a79cb53b1deac7c7a4f7f05a3c4be366769",
    ),
    (
        "9656c1bf1e67a01c6650187223d64b17252b79f099beb3f02ff92b3d3fecdec8",
        "f77d51b3c665b103c330cc5a0e572febf302ed8582ffb926924317d1877cdf03",
        "9efdf546737623a002bd6603e945647a320271fa278a114c9e1e3309fedf6d6b",
    ),
    (
        "0f54ef0b7c348f7d45ba2791f378f3779c1b5ee52f7b96fe6889772a44efe985",
        "932c46fba0809c98f980dcddc339d70b5983931cdcf33f6593ebb9f4ba0f3d84",
        "bc02c6b0f6ea3f6310d7993097513a2423a2e25830d82df481e33af62a59b36b",
    ),
    (
        "4f0d1457cb76849de6fc8e452e117deed77e5358f650f45663881282145eaa18",
        "dafdbbbd8e1387f75bb65488e5a2e4d2ecb53bf0e6d637b7555036ba5cab02b0",
        "4c33a8082327ad8c96476a2356875d53192f0f1019cdf19ef8f08fa694a7c69c",
    ),
)


@dataclass(frozen=True)
class RunSpec:
    name: str
    log_sha256: str
    engine_pid: int
    glog_date: str
    app_date: str
    code_pin: str
    marker: str
    expected_fingerprints: tuple[tuple[str, str, str], ...]


RUN_SPECS = (
    RunSpec(
        "accepted_unobserved",
        "c680eb587886872094e09824fd0080a65b151b132a3547bb1bb447dfffe52f51",
        53458,
        "I0807",
        "08-07",
        "b3c25df47ac9",
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE",
        ACCEPTED_FINGERPRINTS,
    ),
    RunSpec(
        "historical_dense_boundary_callback",
        "29911df0255b8214524f00554650e3dd3b782291c8dba75252d751dd5c6b8f12",
        327160,
        "I0814",
        "08-14",
        "4e3aa9666cef",
        "greenfield_legacy_layer0_dense_boundary_p8155_20260814T134610676758377Z",
        CALLBACK_FINGERPRINTS,
    ),
    RunSpec(
        "current_layer1_rms_input_callback",
        "2285a8938acd82b4b70e274181aa9b645f1eb9e143e69a8514ec37c21e23f549",
        1618227,
        "I0829",
        "08-29",
        "8dc7d20fedca",
        "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z",
        CALLBACK_FINGERPRINTS,
    ),
)

_FINGERPRINT_RE = re.compile(
    r"\(HLO module jit_step_fun_impl\): "
    r"(?P<kind>Executable fingerprint \(including data segments\)|"
    r"Executable fingerprint|Host transfer fingerprint):"
    r"(?P<digest>[0-9a-f]{64})"
)
_BUCKET_RE = re.compile(
    r"Compilation of worker0 backbone --> \{'num_tokens': (?P<tokens>\d+), "
    r"'num_reqs': 1\} finished"
)
_FORBIDDEN_INVENTORY_TOKEN = re.compile(
    r"(?:^|[/_.-])(?:stablehlo|hlo|xla|compile|compiled|executable|fingerprint)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)


def _read_exact(path: Path, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise BenchmarkValidationError(f"sealed callback evidence drifted: {path}")
    return raw


def _append_distinct(values: list[Any], value: Any) -> None:
    if not values or values[-1] != value:
        values.append(value)


def extract_run_fingerprints(raw: bytes, spec: RunSpec) -> dict[str, Any]:
    """Extract one anchored run; same-day or embedded runs cannot enter it."""

    if sha256(raw).hexdigest() != spec.log_sha256:
        raise BenchmarkValidationError(f"{spec.name} log SHA drifted")
    text = raw.decode("utf-8", errors="strict")
    pid_marker = f"(EngineCore pid={spec.engine_pid})"
    lines = text.splitlines()
    code_marker = f"GLM_CODE_FINGERPRINT: git={spec.code_pin} dirty=0"
    code_lines = [
        line
        for line in lines
        if pid_marker in line
        and f"INFO {spec.app_date} " in line
        and "GLM_CODE_FINGERPRINT:" in line
    ]
    if spec.marker not in text or not code_lines:
        raise BenchmarkValidationError(f"{spec.name} run anchors are missing")
    if any(code_marker not in line for line in code_lines):
        raise BenchmarkValidationError(f"{spec.name} code fingerprint drifted")
    config_lines = [
        line
        for line in lines
        if pid_marker in line
        and f"INFO {spec.app_date} " in line
        and "Initializing a V1 LLM engine" in line
        and "compilation_config=" in line
    ]
    if len(config_lines) != 1 or "'debug_dump_path': None" not in config_lines[0]:
        raise BenchmarkValidationError(f"{spec.name} debug-dump configuration drifted")

    raw_records: list[tuple[str, str]] = []
    buckets: list[int] = []
    for line in lines:
        if pid_marker not in line:
            continue
        if spec.glog_date in line:
            match = _FINGERPRINT_RE.search(line)
            if match is not None:
                raw_records.append((match.group("kind"), match.group("digest")))
        if f"INFO {spec.app_date} " in line:
            match = _BUCKET_RE.search(line)
            if match is not None:
                _append_distinct(buckets, int(match.group("tokens")))

    expected_kinds = (
        "Executable fingerprint",
        "Executable fingerprint (including data segments)",
        "Host transfer fingerprint",
    )
    if len(raw_records) % len(expected_kinds):
        raise BenchmarkValidationError(f"{spec.name} fingerprint triple count drifted")
    observed: list[tuple[str, str, str]] = []
    for offset in range(0, len(raw_records), len(expected_kinds)):
        group = raw_records[offset : offset + len(expected_kinds)]
        if tuple(kind for kind, _ in group) != expected_kinds:
            raise BenchmarkValidationError(
                f"{spec.name} fingerprint triple order drifted"
            )
        triple = tuple(digest for _, digest in group)
        _append_distinct(observed, triple)
    observed_triples = tuple(observed)
    if tuple(buckets) != TOKEN_BUCKETS:
        raise BenchmarkValidationError(f"{spec.name} token-bucket order drifted")
    if observed_triples != spec.expected_fingerprints:
        raise BenchmarkValidationError(f"{spec.name} fingerprint order drifted")
    return {
        "app_date": spec.app_date,
        "code_pin": spec.code_pin,
        "engine_pid": spec.engine_pid,
        "fingerprints": [
            {
                "executable": executable_digest,
                "executable_including_data_segments": data_digest,
                "host_transfer": host_digest,
                "num_tokens": tokens,
            }
            for tokens, (executable_digest, data_digest, host_digest) in zip(
                TOKEN_BUCKETS, observed_triples, strict=True
            )
        ],
        "glog_date": spec.glog_date,
        "log_sha256": spec.log_sha256,
        "name": spec.name,
        "run_marker": spec.marker,
    }


def validate_accepted_oracle(manifest_raw: bytes, success_raw: bytes) -> dict[str, Any]:
    """Authenticate the accepted DB485 oracle and its terminal seal."""

    if sha256(manifest_raw).hexdigest() != ACCEPTED_MANIFEST_FILE_SHA256:
        raise BenchmarkValidationError("accepted oracle manifest bytes drifted")
    if sha256(success_raw).hexdigest() != ACCEPTED_SUCCESS_SHA256:
        raise BenchmarkValidationError("accepted oracle SUCCESS bytes drifted")
    manifest = json.loads(manifest_raw)
    source = manifest.get("source", {})
    semantic_manifest = dict(manifest)
    semantic_manifest.pop("manifest_sha256", None)
    semantic_sha256 = sha256(
        json.dumps(
            semantic_manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if semantic_sha256 != ACCEPTED_MANIFEST_SHA256:
        raise BenchmarkValidationError("accepted oracle semantic manifest drifted")
    expected_manifest_fields = {
        "artifact_kind": "greenfield_short_context_legacy_dsa_oracle",
        "capture_code_hash": ACCEPTED_SUCCESS_FIELDS["sealer_code_hash"],
        "legacy_repository_pin_at_capture": ACCEPTED_FULL_CODE_PIN,
        "manifest_sha256": ACCEPTED_MANIFEST_SHA256,
        "source_capture_code_hash": ACCEPTED_SUCCESS_FIELDS["source_capture_code_hash"],
    }
    if any(
        manifest.get(key) != value for key, value in expected_manifest_fields.items()
    ):
        raise BenchmarkValidationError("accepted oracle manifest contract drifted")
    expected_source_fields: dict[str, Any] = {
        "benchmark": "passkey_L8192_d0.5",
        "fork_git": "b3c25df47",
        "item_id": "t0",
        "item_row_id": 1769,
        "process_count": 8,
        "run_id": 485,
        "source_row_sha256": (
            "c8ae54b36ff3ce79608766cebf5fa6046dc401a0884cdc4b10a917dd4098e996"
        ),
    }
    if any(source.get(key) != value for key, value in expected_source_fields.items()):
        raise BenchmarkValidationError("accepted oracle source identity drifted")
    if len(manifest.get("source_dump_files", ())) != 294:
        raise BenchmarkValidationError("accepted oracle source dump count drifted")

    success_lines = success_raw.decode("utf-8", errors="strict").splitlines()
    if len(success_lines) != len(ACCEPTED_SUCCESS_FIELDS):
        raise BenchmarkValidationError("accepted oracle SUCCESS field count drifted")
    success: dict[str, str] = {}
    for line in success_lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in success:
            raise BenchmarkValidationError("accepted oracle SUCCESS format drifted")
        success[key] = value
    if success != ACCEPTED_SUCCESS_FIELDS:
        raise BenchmarkValidationError("accepted oracle SUCCESS contract drifted")
    if success["manifest_sha256"] != manifest["manifest_sha256"]:
        raise BenchmarkValidationError("accepted oracle manifest link drifted")
    return {
        "artifact_kind": manifest["artifact_kind"],
        "full_code_pin": manifest["legacy_repository_pin_at_capture"],
        "item_id": source["item_id"],
        "item_row_id": source["item_row_id"],
        "manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
        "manifest_sha256": manifest["manifest_sha256"],
        "remote_prefix": success["remote_prefix"],
        "run_id": source["run_id"],
        "success_sha256": ACCEPTED_SUCCESS_SHA256,
    }


def _inventory(
    root: Path, *, prefix: str, expected_count: int, expected_sha: str
) -> dict[str, Any]:
    if not root.is_dir():
        raise BenchmarkValidationError(f"callback archive root is missing: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    listing = "".join(
        f"{prefix}{path.relative_to(root).as_posix()}\n" for path in paths
    )
    if (
        len(paths) != expected_count
        or sha256(listing.encode()).hexdigest() != expected_sha
    ):
        raise BenchmarkValidationError(f"callback archive inventory drifted: {root}")
    forbidden = [
        line for line in listing.splitlines() if _FORBIDDEN_INVENTORY_TOKEN.search(line)
    ]
    if forbidden:
        raise BenchmarkValidationError(
            "callback archive unexpectedly contains a compiler-artifact-named object: "
            f"{forbidden[0]}"
        )
    return {
        "file_count": len(paths),
        "forbidden_compiler_artifact_count": 0,
        "listing_prefix": prefix,
        "listing_sha256": expected_sha,
    }


def classify_callback_executable_class(
    accepted_root: Path,
    *,
    historical_root: Path,
    current_root: Path,
    rejection_artifact_path: Path,
) -> dict[str, Any]:
    accepted_oracle = validate_accepted_oracle(
        (accepted_root / "oracle/manifest.json").read_bytes(),
        (accepted_root / "SUCCESS").read_bytes(),
    )
    accepted_log = accepted_root / "source_capture/legacy.log"
    historical_log = historical_root / (
        "diagnostic_local/"
        "greenfield_legacy_layer0_dense_boundary_p8155_20260814T134610676758377Z/legacy.log"
    )
    current_log = current_root / (
        "diagnostic_local/"
        "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z/legacy.log"
    )
    runs = [
        extract_run_fingerprints(path.read_bytes(), spec)
        for path, spec in zip(
            (accepted_log, historical_log, current_log), RUN_SPECS, strict=True
        )
    ]

    rejection_raw = _read_exact(rejection_artifact_path, REJECTION_ARTIFACT_SHA256)
    rejection = json.loads(rejection_raw)
    positions = rejection.get("dsa_comparison", {}).get("selected_positions", {})
    scores = rejection.get("dsa_comparison", {}).get("selected_scores", {})
    repeated = rejection.get("repeated_failure_class", {})
    protected = rejection.get("protected_run", {})
    if (
        rejection.get("classification") != "REJECTED_OBSERVER_PERTURBATION"
        or positions.get("mismatch_count") != 557_434
        or scores.get("mismatch_count") != 573_438
        or protected.get("passkey_accuracy_percent") != 100.0
        or repeated.get("historical_db") != 551
        or not all(
            repeated.get(field) is True
            for field in (
                "same_first_event",
                "same_position_mismatch_count",
                "same_score_mismatch_count",
            )
        )
    ):
        raise BenchmarkValidationError("callback DSA rejection contract drifted")

    accepted_set = set(ACCEPTED_FINGERPRINTS)
    callback_set = set(CALLBACK_FINGERPRINTS)
    accepted_executable = {pair[0] for pair in ACCEPTED_FINGERPRINTS}
    callback_executable = {pair[0] for pair in CALLBACK_FINGERPRINTS}
    accepted_including_data = {triple[1] for triple in ACCEPTED_FINGERPRINTS}
    callback_including_data = {triple[1] for triple in CALLBACK_FINGERPRINTS}
    accepted_host = {triple[2] for triple in ACCEPTED_FINGERPRINTS}
    callback_host = {triple[2] for triple in CALLBACK_FINGERPRINTS}
    if (
        accepted_set & callback_set
        or accepted_executable & callback_executable
        or accepted_including_data & callback_including_data
        or accepted_host & callback_host
        or runs[1]["fingerprints"] != runs[2]["fingerprints"]
    ):
        raise BenchmarkValidationError("callback executable-class relation drifted")

    inventories = {
        "accepted_unobserved": _inventory(
            accepted_root,
            prefix="",
            expected_count=509,
            expected_sha="034ef89493de7bbc275c0a93ebf13cce6f676fb81a6d2e1659241348139bbd78",
        ),
        "historical_dense_boundary_callback": _inventory(
            historical_root,
            prefix=(
                "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/dense_boundary/8k/"
                "greenfield_legacy_layer0_dense_boundary_p8155_20260814T134610676758377Z/"
            ),
            expected_count=507,
            expected_sha="842b1ccc80eac246c4fcb6db7f78207aec2f1f336dd049821a3ee25829da4e1a",
        ),
        "current_layer1_rms_input_callback": _inventory(
            current_root,
            prefix=(
                "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/layer1_rms_input/8k/"
                "greenfield_legacy_layer1_rms_input_p8155_20260829T192233297523063Z/"
            ),
            expected_count=45,
            expected_sha="4f53e8feb6f234f5fd00dd11724ea09337ecaa4319502bcadbf89f011b312a17",
        ),
    }
    return {
        "artifact_kind": "greenfield_callback_executable_class_certificate",
        "accepted_oracle": accepted_oracle,
        "classification": CLASSIFICATION,
        "compiler_artifact_inventory": inventories,
        "dsa_rejection": {
            "event_0_exact": True,
            "historical_db": 551,
            "passkey_accuracy_percent": 100.0,
            "position_mismatch_count": 557_434,
            "rejection_artifact_sha256": REJECTION_ARTIFACT_SHA256,
            "score_mismatch_count": 573_438,
        },
        "fingerprint_relations": {
            "accepted_callback_triple_intersection_count": 0,
            "accepted_callback_executable_intersection_count": 0,
            "accepted_callback_executable_including_data_intersection_count": 0,
            "accepted_callback_host_transfer_intersection_count": 0,
            "accepted_triple_count": len(accepted_set),
            "callback_triple_count": len(callback_set),
            "future_admissibility_identity_count": 3,
            "historical_equals_current_in_bucket_order": True,
        },
        "limitations": [
            "No retained inventory contains a compiler-artifact-named object.",
            "Object payloads were not exhaustively scanned for embedded compiler IR.",
            "Fingerprint class identity does not localize an optimized operation.",
            "The accepted and callback source pins contain diagnostic-support deltas beyond one callback.",
            "Materialization, fusion, scheduling, and collective-order mechanisms remain unproven.",
            "A CPU JAXpr or StableHLO is not the missing TPU optimized executable.",
            "Compiler fingerprints are treated as executable-class identities, not as semantic proof.",
            "The surviving-cache inspection was a dated host observation, not a sealed artifact.",
        ],
        "runs": runs,
        "status": "OFFLINE_EVIDENCE_ONLY_NO_GATE_D_OR_EXECUTION_CLAIM",
        "tombstone_pin": TOMBSTONE_PIN,
    }


def serialize_callback_executable_class(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
