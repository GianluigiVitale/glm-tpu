"""Fixed local-merge challenger of DB598; not full-model performance admission."""

from hashlib import sha256
import json
from pathlib import Path

from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal
from scripts.greenfield.prefill_window_evidence import same_json

KERNEL = "ws32_prefill_sorted_merge"
PROTOCOL = "ws32-prefill-long-prefix-sorted-local-merge-v1"
PROFILE = "ws32-dsa-sorted-local-key512-default-paired-v1"
REPO = Path(__file__).resolve().parents[2]
BASELINE_TAG = "greenfield_fp8_ws32_prefill_budget_baseline_20260909T021350314569852Z"
BASELINE_SUMMARY_SHA = (
    "5ff75c40fa142e528368c808dd1d7ee30e3024c0ecd0d256a763a4a8cc158969"
)
BINDINGS = {
    "configs/prefill-performance-targets-v1.json": "5f7b99ce09154dfda12258128b6fa96a7cb157b74323424543676c5b473aedba",
    "docs/greenfield/PREFILL_PERFORMANCE_TARGETS.md": "0a8d99ac497a6b0722d744423d73d487be32064c44f0fda2819d74047af70fa4",
    "docs/artifacts/prefill-missing-budget-sealed-20260909.json": "5a249e3d48a2455fc4730a44bd564c514baa7543e85c24e7c1f3649d03885354",
}
# Actual named input shardings matter. Offline TPU-target lowering with the
# flag False reproduces BOTH DB598 originals, byte for byte. No TPU compilation
# or allocation is implied by this preregistration of the True graphs.
ORIGINAL_SHA = {
    "dsa_c131072": "12c12908fb0ce356afbb05dcab39b6121ad396eb9cb400ffed0616e1b27a0fff",
    "dsa_c262656": "6d71b7e2dfa2195f6b9c20fb6035fcc4aaa69ecea9dfbca9c335b1022c72ae71",
}
CANDIDATE_SHA = {
    "dsa_c131072": "0ef5b6e6d03303d7b330a7673dd2869078a85bf89b90d48a78420b55a1f72016",
    "dsa_c262656": "7e1e0f3d24879434d4820268251a1072cd660fd73c22ad80faadb74d7132c39e",
}
NOTE = (
    "Weight-free exact local sorted-pair merge versus sealed DB598, B32/key512/"
    "top2048/default scorer and unchanged expert8/global merge. Six unprofiled "
    "synthetic cases, no repeated cache overhead, XPlane or model/TTFT claim. "
    "Measurement completion is separate from candidate selection success."
)


class SortedMergeJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_prefill_sorted_merge_journal"

    def _check_identity(self, identity):
        if (
            identity.get("protocol") != PROTOCOL
            or identity.get("profile") != PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("unregistered sorted-merge journal identity")


def registration() -> dict:
    """Same small committed target/baseline bindings on controller and workers."""
    for name, digest in BINDINGS.items():
        path = REPO / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"sorted-merge registration bytes differ: {name}")
    return dict(
        protocol=PROTOCOL,
        profile=PROFILE,
        bound_files=dict(BINDINGS),
        baseline_db_run_id=598,
        baseline_summary_sha256=BASELINE_SUMMARY_SHA,
        sorted_local_merge=True,
        overhead_remeasured=False,
    )


def inspect_program(name: str, stable: str, optimized: str, memory: dict) -> dict:
    from scripts.greenfield import prefill_budget_worker as baseline

    if (
        name not in CANDIDATE_SHA
        or sha256(stable.encode()).hexdigest() != CANDIDATE_SHA[name]
    ):
        raise ValueError("sorted-merge candidate preregistered StableHLO differs")
    result = baseline.inspect_program(name, stable, optimized, memory)
    return dict(result, profile=PROFILE, sorted_local_merge=True)


def baseline_wall() -> dict:
    """Controller-only: rederive DB598 aligned samples from its sealed bytes."""
    from scripts.greenfield import prefill_budget_evidence as evidence

    registration()
    path = Path("/home/gianl/glm-run") / BASELINE_TAG / "summary.json"
    raw = path.read_bytes()
    if path.is_symlink() or sha256(raw).hexdigest() != BASELINE_SUMMARY_SHA:
        raise ValueError("sorted-merge retained DB598 summary differs")
    summary = json.loads(raw)
    if summary["results_db_run_id"] != 598:
        raise ValueError("sorted-merge baseline DB identity differs")
    result = evidence.fleet_wall(summary["runner"]["workers"])
    same_json(result, summary["runner"]["dsa_wall"], "DB598 aligned original samples")
    return result


def compare(actual: dict) -> dict:
    from scripts.greenfield import prefill_budget_probe as probe

    baseline = baseline_wall()
    ratios = {}
    endpoint, midpoint = [], []
    for case in probe.cases():
        a, b = actual["cases"][case.name], baseline["cases"][case.name]
        ratios[case.name] = dict(
            baseline_p50_seconds=b["p50_seconds"],
            candidate_p50_seconds=a["p50_seconds"],
            p50_ratio=a["p50_seconds"] / b["p50_seconds"],
            p99_ratio=a["p99_seconds"] / b["p99_seconds"],
        )
        ratio = ratios[case.name]["p50_ratio"]
        if case.last_valid_length == case.prompt_length:
            endpoint.append(ratio <= 0.9)
        elif case.last_valid_length == case.prompt_length // 2:
            midpoint.append(ratio <= 1.05)
    return dict(
        baseline_db_run_id=598,
        baseline_summary_sha256=BASELINE_SUMMARY_SHA,
        registration=registration(),
        cases=ratios,
        candidate_selection_passed=all(endpoint) and all(midpoint),
        statistical_confidence_claim=False,
        model_performance_claim=False,
        trace_claim=False,
    )
