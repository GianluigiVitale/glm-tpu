"""Spec §23.5: the long-context correctness contract (L7 passkey, L8 e0)."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/greenfield/run_short_decoder_ws32.py"
SEALER = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"


def _events(width: int = 8, events: int = 3, live: int = 3):
    positions = np.full((events, 1, width), -1, dtype=np.int32)
    scores = np.full((events, 1, width), -np.inf, dtype=np.float32)
    counts = np.zeros((events, 1), dtype=np.int32)
    chosen = [10, 4, 7][:live]
    values = [0.9, 0.5, 0.5][:live]
    order = sorted(range(live), key=lambda i: (-values[i], chosen[i]))
    for event in range(events):
        positions[event, 0, :live] = [chosen[i] for i in order]
        scores[event, 0, :live] = [values[i] for i in order]
        counts[event, 0] = live
    return positions, counts, scores, np.arange(events, dtype=np.int32)


def test_within_engine_contract_accepts_a_conforming_observation() -> None:
    from glm_tpu.greenfield.validation import compare_ws32_dsa_within_engine

    positions, counts, scores, ids = _events()
    result = compare_ws32_dsa_within_engine(
        producer_layer_ids=ids, selected_positions=positions,
        selected_valid_counts=counts, selected_scores=scores,
        decode_position=100, step=0, expected_producer_layer_ids=ids,
    )
    assert result["passed"] is True
    assert result["cross_oracle"] is False
    assert result["contract"] == "within_engine_only"


def test_within_engine_contract_refuses_each_violation() -> None:
    """It is not a formality: each of these is a real device-contract failure."""
    from glm_tpu.greenfield.validation import compare_ws32_dsa_within_engine

    positions, counts, scores, ids = _events()

    def check(**overrides):
        arguments = dict(
            producer_layer_ids=ids, selected_positions=positions,
            selected_valid_counts=counts, selected_scores=scores,
            decode_position=100, step=0, expected_producer_layer_ids=ids,
        )
        arguments.update(overrides)
        return compare_ws32_dsa_within_engine(**arguments)["passed"]

    swapped = positions.copy()
    swapped[0, 0, 0], swapped[0, 0, 1] = swapped[0, 0, 1], swapped[0, 0, 0]
    assert check(selected_positions=swapped) is False, "tie order must be canonical"

    dirty = scores.copy()
    dirty[1, 0, 5] = 0.0
    assert check(selected_scores=dirty) is False, "the padding tail must be the sentinel"

    assert check(decode_position=5) is False, "positions must be in range"

    repeated = positions.copy()
    repeated[0, 0, 1] = repeated[0, 0, 0]
    assert check(selected_positions=repeated) is False, "positions must be distinct"

    assert check(expected_producer_layer_ids=np.asarray([0, 1, 9], np.int32)) is False

    overflow = counts.copy()
    overflow[0, 0] = 99
    assert check(selected_valid_counts=overflow) is False


def _runner():
    specification = importlib.util.spec_from_file_location("ws32_runner_lc", RUNNER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_the_long_context_token_result_never_claims_raw_token_exactness() -> None:
    """§23.5 forbids it: the legacy stored TEXT at these lengths, not ids."""

    class _Oracle:
        kind = "e0"
        generated_token_ids = np.asarray([1, 2, 3], dtype=np.int32)
        gold = None

    result = _runner()._long_context_token_result([1, 2, 3], _Oracle(), tokenizer_root=None)
    assert "exact_prefix_match" not in result
    assert result["passkey_matches_gold"] is None
    assert result["legacy_diagnostic"]["legacy_ids_match"] is True
    assert "diagnostic" in result["legacy_diagnostic"]["note"]


def test_the_runner_refuses_an_adjudication_record_in_long_context_mode() -> None:
    """There is nothing at these lengths for a §21.2 record to adjudicate."""
    source = RUNNER.read_text(encoding="utf-8")
    assert "WS32 long-context runs bind no §21.2 adjudication record" in source
    assert "WS32 long-context seals bind no §21.2 adjudication record" in (
        SEALER.read_text(encoding="utf-8")
    )


def test_the_sealer_classification_makes_no_cross_oracle_claim() -> None:
    """PASSKEY_EXACT/NO_CROSS_ORACLE, and never RAW_TOKENS_EXACT."""
    source = SEALER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"PASSKEY_EXACT", "NO_CROSS_ORACLE", "NO_CORRECTNESS_ORACLE"} <= literals

    start = source.index('basis = ["DSA_WITHIN_ENGINE_EXACT"')
    end = source.index('summary["classification"]', start)
    block = source[start:end]
    assert "RAW_TOKENS_EXACT" in block, "the short-context branch must still be present"
    long_branch = block[: block.index("else:")]
    assert "RAW_TOKENS_EXACT" not in long_branch
    assert "DSA_CROSS_ORACLE_EXACT_ALL_EVENTS" not in long_branch


def test_the_sealer_recomputes_the_runners_own_token_rule() -> None:
    """Two copies of a correctness criterion drift; the sealer imports one."""
    source = SEALER.read_text(encoding="utf-8")
    start = source.index("def _long_context_token_result(")
    end = source.index("def _rank0_dsa_arrays(")
    body = source[start:end]
    assert "run_short_decoder_ws32.py" in body
    assert "_long_context_token_result(" in body


WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"


def _wrapper_profile_table() -> dict[str, dict[str, str]]:
    """The wrapper's ``case "$CONTEXT"`` table, as {label: {var: value}}."""
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index('case "$CONTEXT" in\n  128k_d0_0)')
    block = source[start : source.index("\nesac", start)]
    table: dict[str, dict[str, str]] = {}
    label: str | None = None
    for line in block.splitlines()[1:]:
        stripped = line.strip().rstrip(";;").strip()
        if stripped.endswith(")") and "=" not in stripped:
            label = stripped[:-1]
            table[label] = {}
        elif "=" in stripped and label is not None:
            key, value = stripped.split("=", 1)
            table[label][key] = value
    return table


def test_the_wrapper_binds_exactly_the_enforcement_surfaces_profiles() -> None:
    """A wrapper pin the sealer's registry does not know is not a §23.5 run.

    The wrapper supplies paths; the registry in
    ``glm_tpu/greenfield/validation/long_context_oracle.py`` supplies identity,
    and the sealer refuses an oracle that is not the label's. If the two tables
    ever disagree, a run would be launched against one capture and sealed
    against another's identity, so they are compared here.
    """
    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
    )

    table = _wrapper_profile_table()
    assert set(table) == set(WS32_LONG_CONTEXT_PROFILES)
    for label, entry in WS32_LONG_CONTEXT_PROFILES.items():
        arm = table[label]
        assert arm["LONG_CONTEXT_KIND"] == entry["kind"]
        assert arm["LONG_CONTEXT_PROFILE"] == entry["profile"]
        assert arm["LONG_CONTEXT_MANIFEST_SHA"] == entry["manifest_sha256"]
        assert arm["LONG_CONTEXT_SUCCESS_SHA"] == entry["success_sha256"]
        assert entry["profile"] in arm["LONG_CONTEXT_RUN"]


def test_a_passkey_run_observes_at_least_the_tokens_its_criterion_reads() -> None:
    """The L7 criterion detokenises the legacy twenty; fourteen would truncate it."""
    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
    )

    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("if [[ $LONG_CONTEXT_KIND == passkey ]]; then")
    block = source[start : source.index("fi", start)]
    assert "readonly OBSERVER_STEPS=20" in block
    for entry in WS32_LONG_CONTEXT_PROFILES.values():
        if entry["kind"] == "passkey":
            assert entry["generated_token_count"] <= 20


def test_the_wrapper_never_names_a_short_context_oracle_at_these_lengths() -> None:
    """§23.5: the short-context pins are declared vacant, not merely unused."""
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index('if [[ -n $LONG_CONTEXT_KIND ]]; then\n  readonly ORACLE_CLI=')
    long_arm = source[start : source.index("else", start)]
    assert "--token-oracle-dir" not in long_arm
    assert "--dsa-oracle-dir" not in long_arm
    assert long_arm.count('"$ZERO_SHA"') == 4


def test_the_registry_refuses_an_oracle_that_is_not_the_labels_capture() -> None:
    """A label must not be sealable from another depth's sealed capture."""
    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
        require_ws32_long_context_profile,
    )

    class _Oracle:
        def __init__(self, entry: dict) -> None:
            self.kind = entry["kind"]
            self.depth = entry["depth"]
            self.prompt_token_ids = np.zeros(entry["prompt_token_count"], np.int32)
            self.generated_token_ids = np.zeros(entry["generated_token_count"], np.int32)
            self.source_run_id = entry["source_run_id"]
            self.item_row_id = entry["item_row_id"]
            self.manifest_sha256 = entry["manifest_sha256"]

    entry = WS32_LONG_CONTEXT_PROFILES["128k_d1_0"]
    oracle = _Oracle(dict(entry))
    require_ws32_long_context_profile(
        "128k_d1_0", oracle, success_sha256=entry["success_sha256"]
    )
    with pytest.raises(ValueError):
        require_ws32_long_context_profile(
            "128k_d0_0", oracle, success_sha256=entry["success_sha256"]
        )
    with pytest.raises(ValueError):
        require_ws32_long_context_profile(
            "128k_d1_0", oracle, success_sha256="0" * 64
        )
    with pytest.raises(ValueError):
        require_ws32_long_context_profile(
            "8k", oracle, success_sha256=entry["success_sha256"]
        )


def test_a_long_context_label_cannot_be_sealed_without_its_oracle() -> None:
    """Otherwise a 128K label would seal a run holding a short-context oracle."""
    source = SEALER.read_text(encoding="utf-8")
    assert "WS32 long-context labels require the long-context oracle flags" in source
    assert "WS32 long-context seals bind no short-context oracle" in source
    assert "WS32 long-context runs bind no short-context oracle" in (
        RUNNER.read_text(encoding="utf-8")
    )


def test_a_long_context_run_is_not_labelled_a_capacity_measurement() -> None:
    """§23.3 Step C is the SHORT workload at a long capacity; §23.5 is not that."""
    source = SEALER.read_text(encoding="utf-8")
    start = source.index('"capacity_measurement": (')
    block = source[start : source.index("),", start)]
    assert "long_context is not None" in block


def test_the_passkey_criterion_is_proven_computable_before_the_prefill() -> None:
    """A missing tokenizer must fail in seconds, not after a multi-hour prefill."""
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_require_passkey_tooling" in functions
    main = functions["main"]
    call_lines = [
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_require_passkey_tooling", "_initialize_runtime")
    ]
    assert len(call_lines) >= 2
    preflight = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require_passkey_tooling"
    )
    runtime = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_initialize_runtime"
    )
    assert preflight < runtime


def test_the_passkey_preflight_reproduces_a_sealed_gold() -> None:
    """The preflight is only meaningful if it runs the real detokenise+extract."""
    spec = importlib.util.spec_from_file_location("_ws32_runner_lc", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
    )

    tokenizer_root = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
    if not (tokenizer_root / "tokenizer_config.json").exists():
        pytest.skip("the sealed tokenizer is not mounted here")
    entry = WS32_LONG_CONTEXT_PROFILES["128k_d1_0"]
    oracle_dir = next(
        (
            Path("/home/gianl/gcs-models/oracles/greenfield/glm52/long_context")
            / entry["profile"]
        ).glob("*/oracle")
    )
    from glm_tpu.greenfield.validation.long_context_oracle import (
        load_ws32_long_context_oracle,
    )

    oracle = load_ws32_long_context_oracle(
        oracle_dir,
        expected_manifest_sha256=entry["manifest_sha256"],
        expected_success_sha256=entry["success_sha256"],
        expected_kind=entry["kind"],
    )
    module._require_passkey_tooling(oracle, tokenizer_root)
    broken = type(oracle)(**{**oracle.__dict__, "gold": "000000"})
    with pytest.raises(ValueError):
        module._require_passkey_tooling(broken, tokenizer_root)


def test_the_criterion_loads_the_legacy_extractor_by_path_not_by_name() -> None:
    """``bench`` is a namespace package another loader can already have shadowed.

    ``inspect_long_context_oracle`` executes ``glm_longctx.py`` under a private
    module name for provenance; a later ``import bench.glm_longctx`` can be
    answered from that same import machinery and hand back a module without
    ``extract_passkey``. The criterion must load the pinned file itself.
    """
    source = RUNNER.read_text(encoding="utf-8")
    assert "import bench." not in source
    # Assert the CALLS, not their formatting: every call must name the module
    # and carry the capture's digests.
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_legacy_bench_module"
    ]
    assert len(calls) == 2
    for call in calls:
        assert [ast.literal_eval(arg) for arg in call.args] == ["glm_longctx"]
        assert [kw.arg for kw in call.keywords] == ["pinned_files"]

    from glm_tpu.greenfield.validation import load_legacy_bench_module

    pinned = _pinned_legacy_files()
    module = load_legacy_bench_module("glm_longctx", pinned_files=pinned)
    assert module.__name__ == "greenfield_legacy_bench_glm_longctx"
    assert module.extract_passkey("the key is 891482.") == "891482"
    assert load_legacy_bench_module("glm_longctx", pinned_files=pinned) is module
    with pytest.raises(ValueError):
        load_legacy_bench_module("os", pinned_files=pinned)


def _pinned_legacy_files() -> dict:
    """The legacy bench digests the 128K d1.0 capture recorded."""
    import json

    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
    )

    entry = WS32_LONG_CONTEXT_PROFILES["128k_d1_0"]
    manifest = json.loads(
        (_oracle_dir(entry) / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest["legacy_bench_files"]


def _oracle_dir(entry) -> Path:
    """The sealed capture's directory, or a skip when the mount is absent."""
    root = (
        Path("/home/gianl/gcs-models/oracles/greenfield/glm52/long_context")
        / entry["profile"]
    )
    found = sorted(root.glob("*/oracle"))
    if not found:
        pytest.skip("the sealed long-context oracle mount is absent here")
    return found[0]


def _wrapper_observer_steps(kind: str) -> int:
    """OBSERVER_STEPS the wrapper gives a profile of this kind."""
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("if [[ $LONG_CONTEXT_KIND == passkey ]]; then")
    block = source[start : source.index("fi", start)]
    passkey, other = (
        int(line.split("=")[1])
        for line in block.splitlines()
        if "readonly OBSERVER_STEPS=" in line
    )
    return passkey if kind == "passkey" else other


def test_the_legacy_extractor_must_be_the_captures_own_bytes() -> None:
    """A criterion computed from other bytes is not the legacy criterion."""
    from glm_tpu.greenfield.validation import load_legacy_bench_module

    pinned = _pinned_legacy_files()
    tampered = {
        name: (dict(record, sha256="0" * 64) if name == "extract.py" else record)
        for name, record in pinned.items()
    }
    with pytest.raises(ValueError, match="not the pinned capture"):
        load_legacy_bench_module("glm_longctx", pinned_files=tampered)
    with pytest.raises(ValueError, match="pins no digest"):
        load_legacy_bench_module(
            "glm_longctx",
            pinned_files={
                name: record
                for name, record in pinned.items()
                if name != "provenance.py"
            },
        )
    with pytest.raises(ValueError):
        load_legacy_bench_module("glm_longctx", pinned_files={})


def test_l8_cardinality_is_independent_of_the_legacy_diagnostic_length() -> None:
    """The diagnostic oracle length is not a correctness criterion.

    Exercise the cardinality predicate with a short hypothetical observation;
    actual §23.5 E0 independently requires 256 timed steps (tested below).
    """
    from glm_tpu.greenfield.validation.long_context_oracle import (
        WS32_LONG_CONTEXT_PROFILES,
    )

    entry = WS32_LONG_CONTEXT_PROFILES["256k_e0"]
    assert entry["generated_token_count"] == 256
    observed = 1 + _wrapper_observer_steps("e0") + 2 + 10 + 2
    assert observed < entry["generated_token_count"]

    # Grepping for the guard would pass with the two arms swapped, which would
    # reinstate this P0 for L8 and create it for L7. The sealer's OWN condition
    # is extracted and evaluated instead.
    verdict = _cardinality_verdict()
    passkey = WS32_LONG_CONTEXT_PROFILES["128k_d1_0"]

    class _LC:
        def __init__(self, kind, generated):
            self.kind = kind
            self.generated_token_ids = np.zeros(generated, np.int32)

    e0 = _LC("e0", entry["generated_token_count"])
    l7 = _LC("passkey", passkey["generated_token_count"])
    # A 29-token observation does not fail this diagnostic-cardinality predicate;
    # it is not proof of the separate E0 timed-window contract.
    assert verdict([0] * observed, observed, long_context=e0, oracle=None) is False
    # L7 observes 35 and needs its twenty: passes; starved of them, refuses.
    assert verdict([0] * 35, 35, long_context=l7, oracle=None) is False
    assert verdict([0] * 10, 10, long_context=l7, oracle=None) is True
    # Short context keeps its own bound against the sealed oracle.
    assert verdict([0] * 10, 10, long_context=None, oracle=l7) is True
    assert verdict([0] * 35, 35, long_context=None, oracle=l7) is False
    # A wrong cardinality is still refused in every mode.
    assert verdict([0] * 34, 35, long_context=e0, oracle=None) is True
    assert verdict("not-a-list", 35, long_context=e0, oracle=None) is True


def test_wrapper_runs_the_full_e0_timed_window_without_changing_l7() -> None:
    import os
    import subprocess

    source = WRAPPER.read_text()
    iteration_line = source.index("readonly ITERATIONS=")
    start = source.rfind('case "$CONTEXT" in', 0, iteration_line)
    end = source.index("esac", iteration_line) + len("esac")
    assert start >= 0
    block = source[start:end]
    for context, expected in (("256k_e0", 256), ("128k_d0_0", 10),
                              ("128k_d1_0", 10), ("8k", 10), ("2k", 10)):
        result = subprocess.run(["/bin/bash", "-eu", "-c", block + '\nprintf "%s" "$ITERATIONS"'],
                                env={**os.environ, "CONTEXT": context}, capture_output=True, text=True, check=True)
        assert int(result.stdout) == expected
    assert 262144 + _wrapper_observer_steps("e0") + 2 + 256 + 2 + 1 == 262419
    assert 262419 <= 262656


def _cardinality_verdict():
    """The sealer's raw-token cardinality condition, as a callable.

    Extracted from the sealer's AST by finding the ``if`` whose body raises the
    cardinality refusal, so the test evaluates the shipped expression rather
    than a copy of it.
    """
    tree = ast.parse(SEALER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        raised = node.body[0]
        if not isinstance(raised, ast.Raise):
            continue
        if "raw-token cardinality drifted" not in ast.dump(raised):
            continue
        expression = compile(
            ast.Expression(body=node.test), "<cardinality>", "eval"
        )

        def verdict(observed_token_ids, expected_observed_count, *, long_context, oracle):
            return bool(
                eval(  # noqa: S307 - the sealer's own expression, by construction
                    expression,
                    {"type": type, "list": list, "len": len, "any": any, "int": int},
                    {
                        "observed_token_ids": observed_token_ids,
                        "expected_observed_count": expected_observed_count,
                        "long_context": long_context,
                        "oracle": oracle,
                    },
                )
            )

        return verdict
    raise AssertionError("the sealer's cardinality refusal was not found")


def test_a_long_context_seal_checks_the_enforcement_surface() -> None:
    """The §23.5 rules are source in this tree; a §23.5 seal must check it.

    A long-context run binds no §21.2 record, so before this the runs those
    rules govern were the only ones sealing with the tree unchecked.
    """
    source = SEALER.read_text(encoding="utf-8")
    assert (
        "if args.dsa_adjudication_record is not None or long_context is not None:"
        in source
    )
    start = source.index(
        "if args.dsa_adjudication_record is not None or long_context is not None:"
    )
    block = source[start : source.index("enforcement_surface = None", start)]
    for check in (
        "_require_imports_come_from",
        "_require_clean_worktree",
        "_enforcement_surface_identity",
        "_require_reviewed_enforcement",
    ):
        assert check in block


def test_the_criterions_own_code_is_on_the_enforcement_surface() -> None:
    """The sealer executes the runner's rule, which runs the legacy extractor."""
    from importlib import util as _util

    spec = _util.spec_from_file_location("_ws32_sealer_surface", SEALER)
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    surface = set(module._ENFORCEMENT_SURFACE)
    assert "scripts/greenfield/run_short_decoder_ws32.py" in surface
    assert "bench/glm_longctx.py" in surface
    assert "bench/extract.py" in surface


def test_a_long_context_run_never_reports_a_verified_token_count() -> None:
    """A cardinality independent of any match must not read as verification."""
    source = SEALER.read_text(encoding="utf-8")
    start = source.index('"verified_generated_token_count": (')
    block = source[start : source.index("),", start)]
    assert "legacy_diagnostic" not in block
    condensed = " ".join(block.split())
    assert "None if long_context is not None" in condensed


def test_an_l8_row_records_no_correctness_verdict_at_all() -> None:
    """L8 has no oracle, so the DB row must not say the run was correct.

    Exercised here rather than left to the L8 run itself: the branch is
    reachable only for ``kind != "passkey"`` and no L8 run has been sealed.
    """
    spec = importlib.util.spec_from_file_location("_ws32_sealer_rows", SEALER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    e0 = {
        "long_context": {
            "kind": "e0",
            "depth": None,
            "item_row_id": 1516,
            "manifest_sha256": "9" * 64,
            "source_run_id": 402,
            "success_sha256": "8" * 64,
        },
        "context_capacity": 262656,
    }
    note, item_id, gold, correct, score = module._run_rows(e0)
    assert (correct, score) == (None, None)
    assert item_id == "s23_5_long_context_e0_row1516_cap262656"
    for text in (note, gold):
        assert "NO correctness oracle" in text or "No correctness oracle" in text
        assert "raw-tokens-exact" not in text.replace(
            "Nothing here is a raw-tokens-exact or cross-oracle claim", ""
        )
    passkey = {
        "long_context": dict(e0["long_context"], kind="passkey", depth=1.0, item_row_id=1520),
        "context_capacity": 131072,
    }
    _, item_passkey, _, correct_passkey, score_passkey = module._run_rows(passkey)
    assert (correct_passkey, score_passkey) == (1, 1.0)
    assert item_passkey == "s23_5_long_context_passkey_row1520_cap131072"

    # A short-context row keeps its own verdict and its env stays byte-identical
    # to what rows published before §23.5 carry, or rollback cannot find them.
    short = {"dsa_adjudication": None, "capacity_measurement": None}
    assert module._run_rows(short)[3:] == (1, 1.0)
    base = {
        "checkpoint_manifest_sha256": "1",
        "checkpoint_success_sha256": "2",
        "code_hash": "3",
        "context_label": "8k",
        "dsa_oracle_manifest_sha256": "4",
        "dsa_oracle_success_sha256": "5",
        "mesh_sha256": "6",
        "run_tag": "t",
        "strategy_nd_dense": False,
        "strategy_nd_dense_overlay_manifest_sha256": "0" * 64,
        "token_oracle_manifest_sha256": "7",
        "token_oracle_success_sha256": "8",
        "xla_python_client_mem_fraction": ".95",
    }
    assert "long_context" not in module._run_environment(base)
    assert "long_context" in module._run_environment({**base, **e0})
