"""Fixed first numerical workload; never a long-context or speed promotion.

The acquisition is reused by content, not reclassified as a numerical success.
Baseline source and seven graphs keep their original checks. The distinct paired
profile preregisters its reviewed runtime and two CPU-lowered raw StableHLO texts;
actual optimized structure and allocations are checked in the numerical run.
The other five graphs retain exact original/worker-coordinate-only identities.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .ws32_prefill import BatchedPrefillPlan


SHORT_PROFILE = "ws32_b17_b11_2k_cap8192_v1"
PAIRED_SHORT_PROFILE = "ws32_b17_b11_2k_cap8192_paired_sort_v1"
# Explicit not-yet-compiled marker ONLY for the paired main/tail graphs. It is
# never passed as an actual HLO hash to structural inspection or the archive.
FRESH_OPTIMIZED_MARKER = "0" * 64
PAIRED_REGISTRATION = (
    "docs/artifacts/prefill-paired-short-preregistration-20260908.json"
)
PAIRED_SOURCE_PIN = "8b74a4945565f1cf7fb7c0ebfcfc5e66c464bd8b"
PAIRED_RUNTIME = "glm_tpu/greenfield/runtime/ws32_batched_prefill.py"
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
PAIRED_UNCHANGED_LOCATION_FINGERPRINTS = {
    "cache_probe": "6e5187b9eabf80582bda60705688f113461ecded25cac9fcb0432a55e8e77cbf",
    "decode": "e409ebb17b157b070eca717c5da8b9535e972e67a8c4401e8511cbd6483b9817",
    "exact_materialize": "c3e3bdbc968815b5f23a6dbbbe775245fafe484c7ab693b9bc04eb8d5d24e72e",
    "exact_promote": "c4b70d17f8ef22dc9fa371ef1cfbdeca52fe79d357134fafb4c2df92bf810907",
    "observer": "13314371eb431c92080b745f53c37f553d35d58c5cd8897e2ea57a6f1f40a74e",
}


def profile_is_paired(profile: str) -> bool:
    if profile not in (SHORT_PROFILE, PAIRED_SHORT_PROFILE):
        raise ValueError("short numerical profile is not registered")
    return profile == PAIRED_SHORT_PROFILE


def paired_registration(repo: Path) -> dict[str, Any]:
    record = json.loads((repo / PAIRED_REGISTRATION).read_text())
    if (
        record["profile"] != PAIRED_SHORT_PROFILE
        or record["model_source_pin"] != PAIRED_SOURCE_PIN
        or set(record["graphs"]) != {"prefill_chunk", "prefill_tail"}
        or set(record["model_source_overrides"]) != {PAIRED_RUNTIME}
    ):
        raise ValueError("paired short preregistration schema drifted")
    return record


def short_numerical_identity(*, profile: str = SHORT_PROFILE) -> dict[str, Any]:
    """Shared worker/sealer/DB identity, absent on historical serial runs."""
    from .ws32_prefill import PREFILL_MODE

    paired = profile_is_paired(profile)
    acquisition = dict(code_hash=ACQUISITION_PIN, receipt_sha256=RECEIPT_SHA256)
    if paired:
        # Distinguish an offline registration from an acquired optimized graph.
        acquisition.update(
            variant_source_pin=PAIRED_SOURCE_PIN,
            preregistration_path=PAIRED_REGISTRATION,
            optimized_graphs_acquired_in_numerical_run=True,
        )
    return dict(
        prefill_mode=PREFILL_MODE,
        batched_prefill_profile=profile,
        batched_prefill_plan=SHORT_PLAN.identity(),
        batched_prefill_acquisition=acquisition,
        prefill_memory_reserve_bytes=SHORT_RESERVE_BYTES,
        prefill_budget_seconds=SHORT_BUDGET_SECONDS,
    )


def short_acquisition(repo: Path, *, profile: str = SHORT_PROFILE) -> dict[str, Any]:
    raw = (repo / RECEIPT).read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise ValueError("short prefill acquisition receipt drifted")
    result = json.loads(raw)
    if profile_is_paired(profile):
        for graph, pins in paired_registration(repo)["graphs"].items():
            result["graphs"][graph] = dict(
                stablehlo_sha256=pins["stablehlo_sha256"],
                optimized_hlo_sha256=FRESH_OPTIMIZED_MARKER,
            )
    return result


def require_acquired_model_source(repo: Path, *, profile: str = SHORT_PROFILE) -> None:
    """Read-only working-tree comparison, additional to the worker clean pin."""
    paired = profile_is_paired(profile)
    if paired:
        registration = paired_registration(repo)
        if (
            sha256((repo / PAIRED_RUNTIME).read_bytes()).hexdigest()
            != registration["model_source_overrides"][PAIRED_RUNTIME]
        ):
            raise ValueError(
                "paired numerical runtime source differs from preregistration"
            )
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            PAIRED_SOURCE_PIN if paired else ACQUISITION_PIN,
            "--",
            *MODEL_SOURCE,
            *([":(exclude)" + PAIRED_RUNTIME] if paired else []),
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
    profile_is_paired(profile)
    if (
        plan != SHORT_PLAN
        or type(reserve_bytes) is not int
        or reserve_bytes != SHORT_RESERVE_BYTES
        or type(budget_seconds) not in (int, float)
        or budget_seconds != SHORT_BUDGET_SECONDS
    ):
        raise ValueError(
            "batched numerical input/profile/reserve/budget is not registered"
        )
    if dict(graph_pins) != short_acquisition(repo, profile=profile)["graphs"]:
        raise ValueError("batched numerical requires all seven acquired HLO pairs")


def require_short_numerical_request(
    args: Any, *, prompt_length: int, repo: Path
) -> None:
    """Small pre-load check shared by worker and sealer; no JAX/cloud calls."""
    from .ws32_prefill import PREFILL_MODE, require_batched_profile

    if args.prefill_mode != PREFILL_MODE:
        raise ValueError("short batched request cannot authorize serial mode")
    require_batched_profile(
        args.prefill_mode,
        exact_dsa=args.exact_dsa == 1,
        host_main_rope_table=args.host_main_rope_table == 1,
        block_rows=args.prefill_chunk,
        long_context=args.long_context,
        adjudication_record=args.dsa_adjudication_record,
        adjudication_sha256=args.dsa_adjudication_sha256,
    )
    if args.strategy_nd_dense != 1 or args.rotary_diagnostic != 0:
        raise ValueError("acquired short decode configuration differs")
    require_short_numerical_inputs(
        profile=args.batched_prefill_profile,
        plan=BatchedPrefillPlan(
            prompt_length, args.prefill_chunk, args.context_capacity
        ),
        reserve_bytes=args.prefill_memory_reserve_bytes,
        budget_seconds=args.prefill_budget_seconds,
        graph_pins={
            graph: {form: getattr(args, f"expected_{graph}_{form}") for form in pins}
            for graph, pins in short_acquisition(
                repo, profile=args.batched_prefill_profile
            )["graphs"].items()
        },
        repo=repo,
    )
    require_acquired_model_source(repo, profile=args.batched_prefill_profile)


def require_hlo_pin_request(args: Any, *, compile_only: bool, repo: Path) -> None:
    """Shared actual startup guard; fresh markers require the complete profile.

    Serial/default/acquisition rules remain unchanged. ONLY the two optimized
    paired prefill pins may be vacant after validating the entire registered
    request, including source and all fourteen graph/form identities.
    """
    materializer_names = {
        f"expected_{graph}_{form}_sha256"
        for graph in ("exact_materialize", "exact_promote")
        for form in ("stablehlo", "optimized_hlo")
    }
    active = {
        name: value
        for name, value in vars(args).items()
        if name.startswith("expected_")
        and "hlo_sha256" in name
        and (args.exact_dsa or name not in materializer_names)
    }
    inactive = {
        name: value
        for name, value in vars(args).items()
        if name in materializer_names and not args.exact_dsa
    }
    if any(value != FRESH_OPTIMIZED_MARKER for value in inactive.values()):
        raise ValueError("default WS32 path must keep exact materializer pins vacant")
    if compile_only:
        if getattr(args, "batched_prefill_profile", "") or any(
            value != FRESH_OPTIMIZED_MARKER for value in active.values()
        ):
            raise ValueError(
                "WS32 acquisition requires vacant active HLO pins and no numerical profile"
            )
        return
    allowed = set()
    if getattr(args, "batched_prefill_profile", "") == PAIRED_SHORT_PROFILE:
        require_short_numerical_request(
            args, prompt_length=SHORT_PLAN.prompt_length, repo=repo
        )
        allowed = {
            "expected_prefill_chunk_optimized_hlo_sha256",
            "expected_prefill_tail_optimized_hlo_sha256",
        }
    if {
        name for name, value in active.items() if value == FRESH_OPTIMIZED_MARKER
    } != allowed:
        raise ValueError(
            "WS32 numerical execution requires acquired active HLO pins except registered paired optimized graphs"
        )


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
    paired = profile_is_paired(profile)
    if type(rows) is not int or graph is None:
        raise ValueError("unknown bounded batched HLO profile")
    pins = short_acquisition(repo, profile=profile)["graphs"][graph]
    identity = report.get("source_location_identity")
    actual_pins = dict(pins)
    if paired:
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema_version") != "ws32_paired_short_fresh_optimized_v1"
            or identity.get("profile") != profile
            or identity.get("graph") != graph
            or identity.get("registered_stablehlo_sha256") != pins["stablehlo_sha256"]
            or report.get("paired_position_sort") is not True
        ):
            raise ValueError("paired short fresh graph binding drifted")
        actual_pins["optimized_hlo_sha256"] = identity["raw_optimized_hlo_sha256"]
    elif identity is not None:
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

    paired = profile_is_paired(profile)
    if graph not in WORKER_LOCATION_FINGERPRINTS:
        raise ValueError("unknown short numerical graph/profile")
    pins = short_acquisition(repo, profile=profile)["graphs"][graph]
    if (expected_stable, expected_optimized) != (
        pins["stablehlo_sha256"],
        pins["optimized_hlo_sha256"],
    ):
        raise ValueError("short graph caller must bind acquired raw HLO pins")
    if sha256(stable.encode()).hexdigest() != expected_stable:
        raise ValueError("short graph StableHLO bytes drifted")
    if paired and graph in ("prefill_chunk", "prefill_tail"):
        if expected_optimized != FRESH_OPTIMIZED_MARKER or not optimized.strip():
            raise ValueError(
                "paired graph requires explicit fresh optimized registration"
            )
        return dict(
            schema_version="ws32_paired_short_fresh_optimized_v1",
            profile=profile,
            graph=graph,
            registered_stablehlo_sha256=expected_stable,
            raw_optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
            optimized_identity_scope="FRESH_ACTUAL_STRUCTURAL_CHECKS_NOT_PRIOR_BYTE_IDENTITY",
        )
    if paired:
        from .ws32_hlo_worker_locations import paired_worker_dsa_location_identity

        identity = paired_worker_dsa_location_identity(optimized)
        if (
            identity["paired_location_equivalence_sha256"]
            != PAIRED_UNCHANGED_LOCATION_FINGERPRINTS[graph]
        ):
            raise ValueError(
                "paired unchanged graph changes more than worker/DSA coordinates"
            )
        return {
            **identity,
            "graph": graph,
            "profile": profile,
            "acquisition_code_hash": ACQUISITION_PIN,
            "acquired_optimized_hlo_sha256": expected_optimized,
        }
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


def validate_short_compiled_memory(
    graph: str, memory: Mapping[str, int], *, profile: str, repo: Path
) -> None:
    """Validate actual compiler allocation before the existing all-live budget.

    Changed short graphs retain exact argument/output/alias geometry. Temporary
    and code bounds are preregistered caps, not predictions of actual residency.
    Every actual allocation is still charged by the32-owner memory validator.
    """
    paired = profile_is_paired(profile)
    original = short_acquisition(repo)["fleet"][0]["compiled"]
    if graph not in original:
        raise ValueError("unknown short memory graph")
    expected = original[graph]["memory"]
    if set(memory) != set(expected) or any(
        type(v) is not int or v < 0 for v in memory.values()
    ):
        raise ValueError("short compiler allocation schema drifted")
    if not paired or graph not in ("prefill_chunk", "prefill_tail"):
        if dict(memory) != expected:
            raise ValueError("short compiler allocation differs from acquisition")
        return
    caps = {"temp_size_in_bytes": 1 << 30, "generated_code_size_in_bytes": 256 << 20}
    if any(
        (value > caps[key] if key in caps else value != expected[key])
        for key, value in memory.items()
    ):
        raise ValueError("paired short compiler allocation exceeds registered bounds")
