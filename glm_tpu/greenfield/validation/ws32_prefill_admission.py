"""Fixed first numerical workload; never a long-context or speed promotion.

The acquisition is reused by content, not reclassified as a numerical success.
Baseline source and seven graphs keep their original checks. The distinct paired
profile preregisters its reviewed runtime and two CPU-lowered raw StableHLO texts;
actual optimized structure and allocations are checked in the numerical run.
The other five graphs retain original executable bytes, allowing only reviewed
worker/DSA/WS32/router source coordinates to move in the paired profile.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .ws32_prefill import BatchedPrefillPlan
from .ws32_canonical_prefill_admission import PROFILE as CANONICAL_SHORT_PROFILE
from .ws32_canonical_8k_admission import PROFILE as CANONICAL_8K_PROFILE
from .ws32_delivery_quality import CONTRACT as DELIVERY_CONTRACT, PROFILE as DELIVERY_SHORT_PROFILE

CANONICAL_PROFILES = (CANONICAL_SHORT_PROFILE, CANONICAL_8K_PROFILE, DELIVERY_SHORT_PROFILE)


SHORT_PROFILE = "ws32_b17_b11_2k_cap8192_v1"
PAIRED_SHORT_PROFILE = "ws32_b17_b11_2k_cap8192_paired_sort_v1"
ROLLED_SHORT_PROFILE = "ws32_b128_b114_2k_cap8192_rolled_panels_merge_v1"
FROZEN_8K_PROFILE = "ws32_b128_b114_8k_cap8192_rolled_panels_merge_live91_v1"
FROZEN_LIVE32_PROFILE = "ws32_b128_b114_8k_cap8192_live32_diagnostic_v1"
FROZEN_FIRST_WINDOW_PROFILE = "ws32_b128_8k_cap8192_first128_diagnostic_v1"
FROZEN_PROFILES = (
    FROZEN_8K_PROFILE,
    FROZEN_LIVE32_PROFILE,
    FROZEN_FIRST_WINDOW_PROFILE,
)
FROZEN_FAILURE_RECEIPT = (
    "docs/artifacts/prefill-frozen-own8k-token-refusal-20260909.json"
)
FROZEN_FAILURE_SHA256 = (
    "e8f0c385cd32f04e50f3ea3e6cdfa4a5b700fd23410e70338fdcaedbcd7c3f1b"
)
FROZEN_SOURCE_PIN = "7456bf6433e1dce966670deb252f4c64bbc5f432"
FROZEN_RECEIPT = "docs/artifacts/prefill-rolled-short-db603-sealed-20260909.json"
FROZEN_RECEIPT_SHA256 = (
    "f349a68ded049812f317671a1eb3e5a7a0b4bfe06d33e9824686dd28507b925b"
)
ROLLED_REGISTRATION = (
    "docs/artifacts/prefill-rolled-short-preregistration-20260909.json"
)
ROLLED_SOURCE_PIN = "c672d4cd4063b01dc049d454980a5779d4fa12ef"
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
ROLLED_PLAN = BatchedPrefillPlan(2034, 128, 8192, mlp_window=True)
FROZEN_8K_PLAN = BatchedPrefillPlan(
    8155, 128, 8192, mlp_window=True, tail_graph_rows=114
)
FROZEN_LIVE32_PLAN = BatchedPrefillPlan(
    8155, 128, 8192, mlp_window=True, tail_graph_rows=114, live_block_rows=32
)
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
    "decode": "d735b8f3bf0ee3c5d5e34df3e27ac92004570c90410184174f7e35a4546be9ee",
    "exact_materialize": "c3e3bdbc968815b5f23a6dbbbe775245fafe484c7ab693b9bc04eb8d5d24e72e",
    "exact_promote": "c4b70d17f8ef22dc9fa371ef1cfbdeca52fe79d357134fafb4c2df92bf810907",
    "observer": "2555001cdcd0dbcf8676f95d411cdcabeccdc5491a9cb14e860c44d559e620e6",
}


def profile_is_paired(profile: str) -> bool:
    if profile not in (
        SHORT_PROFILE,
        PAIRED_SHORT_PROFILE,
        ROLLED_SHORT_PROFILE,
        *CANONICAL_PROFILES,
        *FROZEN_PROFILES,
    ):
        raise ValueError("short numerical profile is not registered")
    return profile != SHORT_PROFILE


def profile_is_rolled(profile: str) -> bool:
    profile_is_paired(profile)
    return profile in (ROLLED_SHORT_PROFILE, *CANONICAL_PROFILES, *FROZEN_PROFILES)


def short_context(profile: str) -> str:
    profile_is_paired(profile)
    return "8k" if profile in (*FROZEN_PROFILES, CANONICAL_8K_PROFILE, DELIVERY_SHORT_PROFILE) else "2k"


def short_budget(profile: str) -> float:
    """Prospective diagnostic ceilings, not §25 speed acceptance thresholds.

    8K allows four times the original2K300s ceiling for four times the live
    prompt. This is deliberately above DB603 short-rate projection (~128s),
    which cannot predict truncating DSA cost. No historical budget is changed.
    """
    profile_is_paired(profile)
    if profile == FROZEN_FIRST_WINDOW_PROFILE:
        return 300.0
    return (
        1200.0
        if profile in (*FROZEN_PROFILES, CANONICAL_8K_PROFILE, DELIVERY_SHORT_PROFILE)
        else SHORT_BUDGET_SECONDS
    )


def short_plan(profile: str) -> BatchedPrefillPlan:
    """Resolve exact workload geometry; a plan alone never permits dispatch."""
    profile_is_paired(profile)
    if profile == FROZEN_LIVE32_PROFILE:
        return FROZEN_LIVE32_PLAN
    if profile in (
        FROZEN_8K_PROFILE,
        FROZEN_FIRST_WINDOW_PROFILE,
        CANONICAL_8K_PROFILE,
        DELIVERY_SHORT_PROFILE,
    ):
        return FROZEN_8K_PLAN
    return (
        ROLLED_PLAN
        if profile in (ROLLED_SHORT_PROFILE, CANONICAL_SHORT_PROFILE)
        else SHORT_PLAN
    )


def short_program_options(profile: str) -> dict[str, Any]:
    """Single worker/registration source for every static implementation flag."""
    paired = profile_is_paired(profile)
    rolled = profile_is_rolled(profile)
    return dict(
        paired_position_sort=paired,
        rolled_prefix=rolled,
        expert_panels=rolled,
        sorted_local_merge=rolled,
        key_tile=512 if rolled else 4096,
        **({"canonical_dense": True} if profile in CANONICAL_PROFILES else {}),
    )


def rolled_registration(repo: Path) -> dict[str, Any]:
    """Bind the exact combined recipe, real-layer prerequisite and final targets."""
    record = json.loads((repo / ROLLED_REGISTRATION).read_text())
    plan = dict(
        prompt_length=2034, block_rows=128, context_capacity=8192, mlp_window=True
    )

    # JSON equality is type-sensitive (True must not stand in for integer1).
    def same(a: Any, b: Any) -> bool:
        return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
            b, sort_keys=True, allow_nan=False
        )

    if (
        record.get("artifact_kind") != "ws32_rolled_short_offline_preregistration_v1"
        or record.get("profile") != ROLLED_SHORT_PROFILE
        or record.get("model_source_pin") != ROLLED_SOURCE_PIN
        or not same(record.get("plan"), plan)
        or not same(
            record.get("program_options"), short_program_options(ROLLED_SHORT_PROFILE)
        )
        or set(record.get("graphs", {})) != {"prefill_chunk", "prefill_tail"}
    ):
        raise ValueError("rolled short preregistration schema/recipe drifted")
    expected_evidence = {
        "docs/artifacts/prefill-rolled-retained-layer-db601-sealed-20260909.json": "318dc278d65e45afd640adf063b73c891e71a6d39fc9e34852f32acccc3d191e",
        "docs/artifacts/prefill-paired-short-sealed-20260909.json": "cdafc8016869e17b8e2325c58f28b001249b473598bcc2da7eb2bc9a66e3ad21",
        "docs/greenfield/PREFILL_PERFORMANCE_TARGETS.md": "0a8d99ac497a6b0722d744423d73d487be32064c44f0fda2819d74047af70fa4",
        "configs/prefill-performance-targets-v1.json": "5f7b99ce09154dfda12258128b6fa96a7cb157b74323424543676c5b473aedba",
    }
    if record.get("evidence") != expected_evidence or any(
        sha256((repo / path).read_bytes()).hexdigest() != digest
        for path, digest in expected_evidence.items()
    ):
        raise ValueError("rolled short prerequisite/target evidence drifted")
    if not same(
        record.get("memory"),
        dict(
            max_argument_growth_bytes=4096,
            output_and_alias="exact original B17/B11",
            max_temporary_bytes=1 << 30,
            max_generated_code_bytes=256 << 20,
            all_live_budget_required=True,
            required_reserve_bytes=SHORT_RESERVE_BYTES,
        ),
    ):
        raise ValueError("rolled short memory registration drifted")
    for pins in record["graphs"].values():
        digest = pins.get("stablehlo_sha256")
        if (
            set(pins) != {"stablehlo_sha256", "stablehlo_bytes"}
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest == FRESH_OPTIMIZED_MARKER
            or type(pins.get("stablehlo_bytes")) is not int
            or not 0 < pins["stablehlo_bytes"] <= 32 << 20
        ):
            raise ValueError("rolled short raw graph registration drifted")
    return record


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
    rolled = profile_is_rolled(profile)
    acquisition = dict(code_hash=ACQUISITION_PIN, receipt_sha256=RECEIPT_SHA256)
    if paired:
        # Distinguish an offline registration from an acquired optimized graph.
        acquisition.update(
            variant_source_pin=ROLLED_SOURCE_PIN if rolled else PAIRED_SOURCE_PIN,
            preregistration_path=ROLLED_REGISTRATION if rolled else PAIRED_REGISTRATION,
            optimized_graphs_acquired_in_numerical_run=True,
        )
    if profile in CANONICAL_PROFILES:
        from . import ws32_canonical_prefill_admission as canonical

        acquisition.update(
            variant_source_pin=canonical.SOURCE_PIN,
            preregistration_path=canonical.STRUCTURAL_RECEIPT,
            corrected_compiler_receipt=canonical.COMPILER_RECEIPT,
            corrected_compiler_receipt_sha256=canonical.COMPILER_SHA256,
            structural_receipt_sha256=canonical.STRUCTURAL_SHA256,
            numerical_inheritance=False,
        )
    if profile == CANONICAL_8K_PROFILE:
        from . import ws32_canonical_8k_admission as own8k

        acquisition.update(
            corrected_short_prerequisites=dict(own8k.PREREQUISITES),
            adjudication_record=own8k.RECORD,
            adjudication_record_sha256=own8k.RECORD_SHA256,
        )
    return dict(
        prefill_mode=PREFILL_MODE,
        **({"validation_contract": DELIVERY_CONTRACT}
           if profile == DELIVERY_SHORT_PROFILE else {}),
        **(
            {
                "frozen_completion_baseline": dict(
                    code_hash=FROZEN_SOURCE_PIN,
                    receipt_path=FROZEN_RECEIPT,
                    receipt_sha256=FROZEN_RECEIPT_SHA256,
                    numerical_inheritance=False,
                )
            }
            if profile in (*FROZEN_PROFILES, *CANONICAL_PROFILES)
            else {}
        ),
        **(
            {
                "live_window_diagnostic": dict(
                    diagnostic_only=True,
                    completion_baseline_replacement=False,
                    failed_run_receipt=FROZEN_FAILURE_RECEIPT,
                    failed_run_receipt_sha256=FROZEN_FAILURE_SHA256,
                )
            }
            if profile in (FROZEN_LIVE32_PROFILE, FROZEN_FIRST_WINDOW_PROFILE)
            else {}
        ),
        **(
            {
                "first_window_diagnostic": dict(
                    diagnostic_only=True,
                    completion_baseline_replacement=False,
                    prompt_length=8155,
                    frontier=128,
                    model_calls=5,
                    live_counts=[128, 32, 32, 32, 32],
                    compiled_graphs=[
                        "exact_materialize",
                        "exact_promote",
                        "prefill_chunk",
                    ],
                    numerical_promotion=False,
                    performance_claim=False,
                )
            }
            if profile == FROZEN_FIRST_WINDOW_PROFILE
            else {}
        ),
        batched_prefill_profile=profile,
        batched_prefill_plan=short_plan(profile).identity(),
        **(
            {"batched_prefill_program_options": short_program_options(profile)}
            if rolled
            else {}
        ),
        batched_prefill_acquisition=acquisition,
        prefill_memory_reserve_bytes=SHORT_RESERVE_BYTES,
        prefill_budget_seconds=short_budget(profile),
    )


def short_acquisition(repo: Path, *, profile: str = SHORT_PROFILE) -> dict[str, Any]:
    if profile == DELIVERY_SHORT_PROFILE:
        from .ws32_delivery_quality import require_prerequisites

        require_prerequisites(repo)
    if profile == CANONICAL_8K_PROFILE:
        from .ws32_canonical_8k_admission import require_prerequisites

        require_prerequisites(repo)
    if (
        profile in FROZEN_PROFILES
        and sha256((repo / FROZEN_RECEIPT).read_bytes()).hexdigest()
        != FROZEN_RECEIPT_SHA256
    ):
        raise ValueError("frozen DB603 baseline receipt drifted")
    if (
        profile in (FROZEN_LIVE32_PROFILE, FROZEN_FIRST_WINDOW_PROFILE)
        and sha256((repo / FROZEN_FAILURE_RECEIPT).read_bytes()).hexdigest()
        != FROZEN_FAILURE_SHA256
    ):
        raise ValueError("frozen live-window diagnostic failure receipt drifted")
    raw = (repo / RECEIPT).read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise ValueError("short prefill acquisition receipt drifted")
    result = json.loads(raw)
    if profile_is_paired(profile):
        if profile in CANONICAL_PROFILES:
            from .ws32_canonical_prefill_admission import (
                registration as canonical_registration,
            )

            registration = canonical_registration(repo)
        else:
            registration = (
                rolled_registration(repo)
                if profile_is_rolled(profile)
                else paired_registration(repo)
            )
        for graph, pins in registration["graphs"].items():
            result["graphs"][graph] = dict(
                stablehlo_sha256=pins["stablehlo_sha256"],
                optimized_hlo_sha256=FRESH_OPTIMIZED_MARKER,
            )
    return result


def require_acquired_model_source(repo: Path, *, profile: str = SHORT_PROFILE) -> None:
    """Read-only working-tree comparison, additional to the worker clean pin."""
    paired = profile_is_paired(profile)
    if profile in CANONICAL_PROFILES:
        from .ws32_canonical_prefill_admission import require_source

        require_source(repo)
        return
    rolled = profile_is_rolled(profile)
    if rolled:
        rolled_registration(repo)
    elif paired:
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
            (
                FROZEN_SOURCE_PIN
                if profile in FROZEN_PROFILES
                else (
                    ROLLED_SOURCE_PIN
                    if rolled
                    else PAIRED_SOURCE_PIN if paired else ACQUISITION_PIN
                )
            ),
            "--",
            *MODEL_SOURCE,
            *([":(exclude)" + PAIRED_RUNTIME] if paired and not rolled else []),
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
        plan != short_plan(profile)
        or type(reserve_bytes) is not int
        or reserve_bytes != SHORT_RESERVE_BYTES
        or type(budget_seconds) not in (int, float)
        or budget_seconds != short_budget(profile)
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
    plan = short_plan(args.batched_prefill_profile)
    require_batched_profile(
        args.prefill_mode,
        exact_dsa=args.exact_dsa == 1,
        host_main_rope_table=args.host_main_rope_table == 1,
        block_rows=args.prefill_chunk,
        long_context=args.long_context,
        adjudication_record=args.dsa_adjudication_record,
        adjudication_sha256=args.dsa_adjudication_sha256,
        mlp_window=plan.mlp_window,
        profile=args.batched_prefill_profile,
        repo=repo,
    )
    if args.strategy_nd_dense != 1 or args.rotary_diagnostic != 0:
        raise ValueError("acquired short decode configuration differs")
    require_short_numerical_inputs(
        profile=args.batched_prefill_profile,
        plan=BatchedPrefillPlan(
            prompt_length,
            args.prefill_chunk,
            args.context_capacity,
            mlp_window=plan.mlp_window,
            tail_graph_rows=plan.tail_graph_rows,
            live_block_rows=plan.live_block_rows,
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
    if getattr(args, "batched_prefill_profile", "") in (
        PAIRED_SHORT_PROFILE,
        ROLLED_SHORT_PROFILE,
        *CANONICAL_PROFILES,
        *FROZEN_PROFILES,
    ):
        require_short_numerical_request(
            args,
            prompt_length=short_plan(args.batched_prefill_profile).prompt_length,
            repo=repo,
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
    the profile's inspector on the actual raw graphs immediately first.
    """
    from ..benchmarking.ws32_batched_prefill import UNREGISTERED

    rolled = profile_is_rolled(profile)
    rows = report.get("block_rows")
    graph = dict((rows, graph) for graph, rows in short_plan(profile).graph_rows).get(
        rows
    )
    paired = profile_is_paired(profile)
    if type(rows) is not int or graph is None:
        raise ValueError("unknown bounded batched HLO profile")
    pins = short_acquisition(repo, profile=profile)["graphs"][graph]
    identity = report.get("source_location_identity")
    actual_pins = dict(pins)
    if paired:
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema_version")
            != (
                "ws32_canonical_short_fresh_optimized_v1"
                if profile in CANONICAL_PROFILES
                else (
                    "ws32_rolled_short_fresh_optimized_v1"
                    if rolled
                    else "ws32_paired_short_fresh_optimized_v1"
                )
            )
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
    if rolled:
        from ..benchmarking.ws32_rolled_prefill import CHECKS

        check_names = [name for name, _ in CHECKS]
        structural_profile = "rolled_b128_b114_v1"
        if profile in CANONICAL_PROFILES:
            check_names.append("canonical_dense_proof")
            structural_profile = "canonical_dense_b128_b114_v1"

        same = lambda a, b: json.dumps(
            a, sort_keys=True, allow_nan=False
        ) == json.dumps(b, sort_keys=True, allow_nan=False)
        if (
            report.get("structural_profile") != structural_profile
            or not same(identity.get("program_options"), short_program_options(profile))
            or not same(identity.get("plan"), short_plan(profile).identity())
            or any(
                not same(report.get(k), v)
                for k, v in short_program_options(profile).items()
            )
            or any(
                not isinstance(report.get(name), Mapping)
                or report[name].get("passed") is not True
                for name in check_names
            )
        ):
            raise ValueError(
                "rolled whole-model HLO structural obligations did not pass"
            )
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


def inspect_short_prefill_graph(
    stable: str,
    optimized: str,
    *,
    graph: str,
    profile: str,
    repo: Path,
    expected_stable: str,
    expected_optimized: str,
) -> dict[str, Any]:
    """Shared worker/sealer composition on actual text, not a stored verdict."""
    rows = dict(short_plan(profile).graph_rows).get(graph)
    if rows is None:
        raise ValueError("short prefill inspection requires main or tail")
    identity = short_graph_identity(
        stable,
        optimized,
        graph=graph,
        profile=profile,
        repo=repo,
        expected_stable=expected_stable,
        expected_optimized=expected_optimized,
    )
    if profile in CANONICAL_PROFILES:
        from ..benchmarking.ws32_canonical_prefill_hlo import (
            inspect_ws32_canonical_prefill_hlo,
        )

        inspector, options = inspect_ws32_canonical_prefill_hlo, {}
    elif profile_is_rolled(profile):
        from ..benchmarking.ws32_rolled_prefill import inspect_ws32_rolled_prefill_hlo

        inspector, options = inspect_ws32_rolled_prefill_hlo, {}
    else:
        from ..benchmarking.ws32_batched_prefill import inspect_ws32_batched_prefill_hlo

        inspector, options = inspect_ws32_batched_prefill_hlo, {
            "paired_position_sort": profile_is_paired(profile)
        }
    report = inspector(
        stable,
        optimized,
        block_rows=rows,
        expected_stablehlo_sha256=expected_stable,
        expected_optimized_hlo_sha256=identity["raw_optimized_hlo_sha256"],
        **options,
    )
    return authorize_short_graph(
        {**report, "source_location_identity": identity}, profile=profile, repo=repo
    )


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
            schema_version=(
                "ws32_canonical_short_fresh_optimized_v1"
                if profile in CANONICAL_PROFILES
                else (
                    "ws32_rolled_short_fresh_optimized_v1"
                    if profile_is_rolled(profile)
                    else "ws32_paired_short_fresh_optimized_v1"
                )
            ),
            profile=profile,
            graph=graph,
            **(
                {
                    "program_options": short_program_options(profile),
                    "plan": short_plan(profile).identity(),
                }
                if profile_is_rolled(profile)
                else {}
            ),
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
                "paired unchanged graph changes more than reviewed source coordinates"
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
    if profile_is_rolled(profile):
        if profile in CANONICAL_PROFILES:
            from .ws32_canonical_prefill_admission import (
                registration as canonical_registration,
            )

            registration = canonical_registration(repo)
        else:
            registration = rolled_registration(repo)
        caps = registration["memory"]
        # Token-ID arguments grow, not the persistent weights/state or outputs.
        # Allow <=4KiB aligned argument growth; actual bytes are independently
        # charged together with all other live buffers before every dispatch.
        if not (
            expected["argument_size_in_bytes"]
            <= memory["argument_size_in_bytes"]
            <= expected["argument_size_in_bytes"] + caps["max_argument_growth_bytes"]
            and memory["output_size_in_bytes"] == expected["output_size_in_bytes"]
            and memory["alias_size_in_bytes"] == expected["alias_size_in_bytes"]
            and memory["temp_size_in_bytes"] <= caps["max_temporary_bytes"]
            and memory["generated_code_size_in_bytes"]
            <= caps["max_generated_code_bytes"]
        ):
            raise ValueError(
                "rolled short compiler allocation exceeds registered bounds"
            )
        return
    caps = {"temp_size_in_bytes": 1 << 30, "generated_code_size_in_bytes": 256 << 20}
    if any(
        (value > caps[key] if key in caps else value != expected[key])
        for key, value in memory.items()
    ):
        raise ValueError("paired short compiler allocation exceeds registered bounds")
