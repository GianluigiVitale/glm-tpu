"""Fixed first numerical workload; never a long-context or speed promotion.

The acquisition is reused by content, not reclassified as a numerical success.
Model source must remain the acquired source. Enforcement/worker code may change
under review; all seven StableHLO texts remain byte-exact. Optimized HLO permits
only worker debug-coordinate changes, retaining both original and actual raw hashes.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .ws32_prefill import BatchedPrefillPlan


SHORT_PROFILE = "ws32_b17_b11_2k_cap8192_v1"
ACQUISITION_PIN = "133fe71fff18c6514ab76ca89d432a90c03b01dd"
RECEIPT = "docs/artifacts/prefill-batched-seven-graph-acquisition-20260908.json"
RECEIPT_SHA256 = "25f322241a66c1747d111513c0a8f91981a461cf319204984e5e47d816a2a2fc"
SHORT_PLAN = BatchedPrefillPlan(2034, 17, 8192)
# A numerical experiment ceiling, NOT a performance acceptance target. The
# historical serial2K estimate is ~237s; a 300s ceiling bounds first-test cost.
SHORT_BUDGET_SECONDS = 300.0
# >2x the ~0.4GiB compiled-vs-observed discrepancy at the old long-capacity test.
# New execution must both budget and actually retain this margin on every chip.
SHORT_RESERVE_BYTES = 1 << 30
SHORT_DEVICE_LIMIT_BYTES = 33_014_398_976
WORKER_LOCATION_FINGERPRINTS = {
    "cache_probe": "6e5187b9eabf80582bda60705688f113461ecded25cac9fcb0432a55e8e77cbf",
    "decode": "1f1663e2c2eda29f89ac02e9107a2be67e0000ab0ab8f60668a2d93c4a7e06e4",
    "exact_materialize": "c3e3bdbc968815b5f23a6dbbbe775245fafe484c7ab693b9bc04eb8d5d24e72e",
    "exact_promote": "c4b70d17f8ef22dc9fa371ef1cfbdeca52fe79d357134fafb4c2df92bf810907",
    "observer": "afeb9749366425d9b62590d8669412799785db3198431524a493b671b41ecd75",
    "prefill_chunk": "e0862f7e1dc9cf058a32d7fbf366ad865ccb5435457450f748fc782d74581782",
    "prefill_tail": "acde7c87450f09e6d308af036b5b1c39e256f0b1fad13b1f1ccbe5a1eff1cf4d",
}
MODEL_SOURCE = (
    "glm_tpu/greenfield/runtime",
    "glm_tpu/greenfield/kernels",
    "glm_tpu/greenfield/sharding",
    "glm_tpu/greenfield/types.py",
    "configs/glm-5.2-fp8-config.json",
)


def short_numerical_identity() -> dict[str, Any]:
    """Shared worker/sealer/DB identity, absent on historical serial runs."""
    from .ws32_prefill import PREFILL_MODE

    return dict(
        prefill_mode=PREFILL_MODE, batched_prefill_profile=SHORT_PROFILE,
        batched_prefill_plan=SHORT_PLAN.identity(),
        batched_prefill_acquisition=dict(code_hash=ACQUISITION_PIN, receipt_sha256=RECEIPT_SHA256),
        prefill_memory_reserve_bytes=SHORT_RESERVE_BYTES,
        prefill_budget_seconds=SHORT_BUDGET_SECONDS,
    )


def short_acquisition(repo: Path) -> dict[str, Any]:
    raw = (repo / RECEIPT).read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise ValueError("short prefill acquisition receipt drifted")
    return json.loads(raw)


def require_acquired_model_source(repo: Path) -> None:
    """Read-only working-tree comparison, additional to the worker clean pin."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            ACQUISITION_PIN,
            "--",
            *MODEL_SOURCE,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("batched numerical model source differs from acquisition")


def require_short_numerical_inputs(
    *,
    profile: str,
    plan: BatchedPrefillPlan,
    reserve_bytes: int,
    budget_seconds: float,
    graph_pins: Mapping[str, Mapping[str, str]],
    repo: Path,
) -> None:
    if (
        profile != SHORT_PROFILE
        or plan != SHORT_PLAN
        or type(reserve_bytes) is not int
        or reserve_bytes != SHORT_RESERVE_BYTES
        or type(budget_seconds) not in (int, float)
        or budget_seconds != SHORT_BUDGET_SECONDS
    ):
        raise ValueError(
            "batched numerical input/profile/reserve/budget is not registered"
        )
    if dict(graph_pins) != short_acquisition(repo)["graphs"]:
        raise ValueError("batched numerical requires all seven acquired HLO pairs")


def require_short_numerical_request(args: Any, *, prompt_length: int, repo: Path) -> None:
    """Small pre-load check shared by worker and sealer; no JAX/cloud calls."""
    from .ws32_prefill import PREFILL_MODE, require_batched_profile

    if args.prefill_mode != PREFILL_MODE:
        raise ValueError("short batched request cannot authorize serial mode")
    require_batched_profile(
        args.prefill_mode, exact_dsa=args.exact_dsa == 1,
        host_main_rope_table=args.host_main_rope_table == 1,
        block_rows=args.prefill_chunk, long_context=args.long_context,
        adjudication_record=args.dsa_adjudication_record,
        adjudication_sha256=args.dsa_adjudication_sha256,
    )
    if args.strategy_nd_dense != 1 or args.rotary_diagnostic != 0:
        raise ValueError("acquired short decode configuration differs")
    require_short_numerical_inputs(
        profile=args.batched_prefill_profile,
        plan=BatchedPrefillPlan(prompt_length, args.prefill_chunk, args.context_capacity),
        reserve_bytes=args.prefill_memory_reserve_bytes,
        budget_seconds=args.prefill_budget_seconds,
        graph_pins={
            graph: {form: getattr(args, f"expected_{graph}_{form}") for form in pins}
            for graph, pins in short_acquisition(repo)["graphs"].items()
        },
        repo=repo,
    )
    require_acquired_model_source(repo)


def authorize_short_graph(
    report: Mapping[str, Any], *, profile: str, repo: Path
) -> dict[str, Any]:
    """Apply bounded registration to a freshly rederived structural report.

    Not a standalone verifier of an untrusted report. Worker/sealer must run
    inspect_ws32_batched_prefill_hlo on the actual raw graphs immediately first.
    """
    from ..benchmarking.ws32_batched_prefill import UNREGISTERED

    rows = report.get("block_rows")
    graph = {17: "prefill_chunk", 11: "prefill_tail"}.get(rows)
    if profile != SHORT_PROFILE or type(rows) is not int or graph is None:
        raise ValueError("unknown bounded batched HLO profile")
    pins = short_acquisition(repo)["graphs"][graph]
    identity = report.get("source_location_identity")
    actual_pins = dict(pins)
    if identity is not None:
        if (
            identity.get("worker_location_equivalence_sha256")
            != WORKER_LOCATION_FINGERPRINTS[graph]
            or identity.get("acquired_optimized_hlo_sha256")
            != pins["optimized_hlo_sha256"]
            or identity.get("graph") != graph
        ):
            raise ValueError("bounded batched source-location identity drifted")
        actual_pins["optimized_hlo_sha256"] = identity["raw_optimized_hlo_sha256"]
    if any(report.get(key) != value for key, value in actual_pins.items()):
        raise ValueError("bounded batched HLO differs from acquired original")
    if (
        report.get("violations") != [UNREGISTERED]
        or report.get("passed") is not False
        or report.get("profile_registered") is not False
    ):
        raise ValueError("bounded batched HLO structural checks did not pass")
    return {
        **report,
        "profile_registered": True,
        "profile_name": profile,
        "passed": True,
        "violations": [],
        "numerical_claim": False,
        "performance_claim": False,
        "runtime_memory_admitted": False,
    }


def short_graph_identity(
    stable: str,
    optimized: str,
    *,
    graph: str,
    profile: str,
    repo: Path,
    expected_stable: str,
    expected_optimized: str,
) -> dict[str, Any]:
    """Keep raw SHA plus exact worker-coordinate-only equivalence to acquisition."""
    from .ws32_hlo_worker_locations import worker_location_identity

    if profile != SHORT_PROFILE or graph not in WORKER_LOCATION_FINGERPRINTS:
        raise ValueError("unknown short numerical graph/profile")
    pins = short_acquisition(repo)["graphs"][graph]
    if (expected_stable, expected_optimized) != (
        pins["stablehlo_sha256"],
        pins["optimized_hlo_sha256"],
    ):
        raise ValueError("short graph caller must bind acquired raw HLO pins")
    if sha256(stable.encode()).hexdigest() != expected_stable:
        raise ValueError("short graph StableHLO bytes drifted")
    identity = worker_location_identity(optimized)
    if (
        identity["worker_location_equivalence_sha256"]
        != WORKER_LOCATION_FINGERPRINTS[graph]
    ):
        raise ValueError("short graph changes more than worker debug coordinates")
    return {
        **identity,
        "graph": graph,
        "acquisition_code_hash": ACQUISITION_PIN,
        "acquired_optimized_hlo_sha256": expected_optimized,
    }
