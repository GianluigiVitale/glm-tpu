"""Changed-graph profile never authorizes historical, unpinned or overbudget work."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.validation import ws32_prefill_admission as a
from glm_tpu.greenfield.benchmarking.ws32_batched_prefill import UNREGISTERED
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _expected

ROOT = Path(__file__).resolve().parents[3]
PROFILE = a.PAIRED_SHORT_PROFILE


def test_source_pin_and_original_five_graphs_are_preserved():
    a.require_acquired_model_source(ROOT, profile=PROFILE)
    old = a.short_acquisition(ROOT)
    new = a.short_acquisition(ROOT, profile=PROFILE)
    for graph in old["graphs"]:
        if graph in ("prefill_chunk", "prefill_tail"):
            assert (
                new["graphs"][graph]["optimized_hlo_sha256"] == a.FRESH_OPTIMIZED_MARKER
            )
            assert (
                new["graphs"][graph]["stablehlo_sha256"]
                != old["graphs"][graph]["stablehlo_sha256"]
            )
        else:
            assert new["graphs"][graph] == old["graphs"][graph]
    assert a.short_numerical_identity(profile=PROFILE) != a.short_numerical_identity()


@pytest.mark.parametrize("profile", [a.SHORT_PROFILE, PROFILE])
@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_memory_caps_do_not_waive_schema_or_live_geometry(profile, graph):
    old = a.short_acquisition(ROOT)["fleet"][0]["compiled"][graph]["memory"]
    a.validate_short_compiled_memory(graph, old, profile=profile, repo=ROOT)
    for field, value in old.items():
        bad = {
            **old,
            field: (
                (1 << 31)
                if field in ("temp_size_in_bytes", "generated_code_size_in_bytes")
                else value + 1
            ),
        }
        with pytest.raises(ValueError):
            a.validate_short_compiled_memory(graph, bad, profile=profile, repo=ROOT)
        with pytest.raises(ValueError):
            a.validate_short_compiled_memory(
                graph, {**old, field: float(value)}, profile=profile, repo=ROOT
            )
    if profile == PROFILE:
        a.validate_short_compiled_memory(
            graph,
            {
                **old,
                "temp_size_in_bytes": 1 << 30,
                "generated_code_size_in_bytes": 256 << 20,
            },
            profile=profile,
            repo=ROOT,
        )


def test_paired_identity_never_hides_failed_structural_checks():
    pins = a.short_acquisition(ROOT, profile=PROFILE)["graphs"]["prefill_chunk"]
    identity = dict(
        schema_version="ws32_paired_short_fresh_optimized_v1",
        profile=PROFILE,
        graph="prefill_chunk",
        registered_stablehlo_sha256=pins["stablehlo_sha256"],
        raw_optimized_hlo_sha256="a" * 64,
    )
    report = dict(
        block_rows=17,
        stablehlo_sha256=pins["stablehlo_sha256"],
        optimized_hlo_sha256="a" * 64,
        source_location_identity=identity,
        paired_position_sort=True,
        passed=False,
        profile_registered=False,
        violations=[UNREGISTERED],
    )
    assert a.authorize_short_graph(report, profile=PROFILE, repo=ROOT)["passed"]
    for change in (
        {"paired_position_sort": False},
        {"violations": [UNREGISTERED, "bad cache"]},
        {"stablehlo_sha256": "b" * 64},
        {"optimized_hlo_sha256": a.FRESH_OPTIMIZED_MARKER},
        {"source_location_identity": None},
    ):
        with pytest.raises(ValueError):
            a.authorize_short_graph({**report, **change}, profile=PROFILE, repo=ROOT)
    with pytest.raises(ValueError):
        a.authorize_short_graph(report, profile=a.SHORT_PROFILE, repo=ROOT)


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_marker_only_registered_paired_changed_graphs(graph, monkeypatch):
    # Small identity-bound text fixture; production raw lowering is tested
    # separately. This tests admission mechanics, not optimized arithmetic.
    stable = "raw test fixture"
    optimized = "actual optimized fixture"
    original = a.short_acquisition(ROOT, profile=PROFILE)
    fake = deepcopy(original)
    fake["graphs"][graph]["stablehlo_sha256"] = sha256(stable.encode()).hexdigest()
    monkeypatch.setattr(a, "short_acquisition", lambda *args, **kwargs: fake)
    kwargs = dict(
        graph=graph,
        profile=PROFILE,
        repo=ROOT,
        expected_stable=sha256(stable.encode()).hexdigest(),
        expected_optimized=a.FRESH_OPTIMIZED_MARKER,
    )
    identity = a.short_graph_identity(stable, optimized, **kwargs)
    assert (
        identity["raw_optimized_hlo_sha256"] == sha256(optimized.encode()).hexdigest()
    )
    for change in (
        {"graph": "decode"},
        {"expected_optimized": "a" * 64},
        {"expected_stable": "b" * 64},
    ):
        with pytest.raises(ValueError):
            a.short_graph_identity(stable, optimized, **{**kwargs, **change})
    with pytest.raises(ValueError):
        a.short_graph_identity(stable, "", **kwargs)


@pytest.mark.parametrize("rows", [11, 17])
def test_paired_helper_delta_is_explicit(rows):
    old = _expected(rows)
    new = _expected(rows, paired_position_sort=True)
    assert not new - old
    removed = old - new
    assert removed.pop(("GatherScatterIndicesBitpacked", "s32", (rows, 4096, 2))) == 42
    assert removed.pop(("GatherScatterIndicesBitpacked", "s32", (rows, 16384, 2))) == 42
    if rows == 17:
        assert removed.pop(("ConcatBitcast", "s32", (278528,))) == 20
    assert not removed
