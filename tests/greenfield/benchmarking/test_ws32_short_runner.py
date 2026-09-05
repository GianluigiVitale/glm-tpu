from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/run_short_decoder_ws32.py"
SPEC = importlib.util.spec_from_file_location("ws32_short_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_ws32_short_runner_latency_distribution_is_exact() -> None:
    result = RUNNER._distribution([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "count": 4,
        "maximum_ms": 4.0,
        "mean_ms": 2.5,
        "minimum_ms": 1.0,
        "p50_ms": 2.5,
        "p90_ms": pytest.approx(3.7),
        "p95_ms": pytest.approx(3.85),
        "p99_ms": pytest.approx(3.97),
    }
    with pytest.raises(ValueError, match="requires samples"):
        RUNNER._distribution([])


def test_ws32_short_runner_evidence_is_append_only_and_hashed(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.json"
    RUNNER._atomic_text(record, "one\n")
    with pytest.raises(FileExistsError, match="append-only"):
        RUNNER._atomic_text(record, "two\n")
    trace = tmp_path / "trace"
    payload = trace / "plugins/profile/run/trace.xplane.pb"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"trace")
    assert RUNNER._trace_files(trace) == [
        {
            "byte_count": 5,
            "relative_path": "plugins/profile/run/trace.xplane.pb",
            "sha256": sha256(b"trace").hexdigest(),
        }
    ]
    with pytest.raises(RuntimeError, match="0 XPlane files"):
        RUNNER._trace_files(tmp_path / "empty")
    tensor_path = tmp_path / "runner.rank0.npz"
    tensor_record = RUNNER._atomic_npz(
        tensor_path,
        positions=np.asarray([1, 2], dtype=np.int32),
    )
    assert tensor_record["filename"] == "runner.rank0.npz"
    assert tensor_record["arrays"]["positions"]["shape"] == [2]
    assert tensor_record["arrays"]["positions"]["dtype"] == "int32"
    with pytest.raises(FileExistsError, match="append-only"):
        RUNNER._atomic_npz(
            tensor_path,
            positions=np.asarray([3], dtype=np.int32),
        )


def test_ws32_short_runner_is_default_off_and_independent() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--compile-only" in source
    assert "verify_ws32_runtime_checkpoint(" in source
    assert "compare_ws32_dsa_step(" in source
    assert "--dsa-adjudication-record" in source
    assert "load_ws32_adjudicated_divergence(" in source
    assert '"dsa_adjudication": dsa_adjudication_record' in source
    assert "compare_ws32_raw_tokens(" in source
    assert "validate_ws32_cache_probe(" in source
    assert "jax.profiler.trace(" in source
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION") != _XLA_MEMORY_FRACTION' in source
    assert '"token_oracle_success_sha256"' in source
    assert '"dsa_oracle_success_sha256"' in source
    assert "performance_claim\": False" in source
    assert "tpu_inference" not in source
    assert "vllm" not in source


def test_ws32_short_acquisition_preserves_all_graphs_before_refusal() -> None:
    vacant = {
        "passed": False,
        "violations": sorted(RUNNER._VACANT_HLO_VIOLATIONS),
    }
    graphs = {
        name: dict(vacant)
        for name in ("cache_probe", "decode", "observer", "prefill_chunk", "prefill_tail")
    }
    for report in graphs.values():
        RUNNER._require_graph_authorized(report, compile_only=True)
    RUNNER._require_acquisition_authorized(graphs, exact_dsa=False)
    with pytest.raises(RuntimeError, match="did not preserve"):
        RUNNER._require_acquisition_authorized(
            {name: graphs[name] for name in graphs if name != "prefill_tail"},
            exact_dsa=False,
        )

    graphs["prefill_tail"] = {
        "passed": False,
        "violations": [
            *sorted(RUNNER._VACANT_HLO_VIOLATIONS),
            "structural refusal",
        ],
    }
    with pytest.raises(RuntimeError, match="structural violations"):
        RUNNER._require_acquisition_authorized(graphs, exact_dsa=False)
    with pytest.raises(RuntimeError, match="failed before execution"):
        RUNNER._require_graph_authorized(
            graphs["prefill_tail"], compile_only=False
        )


def test_ws32_wrapper_pins_the_committed_adjudication_record() -> None:
    import hashlib
    import re

    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    sealer = (root / "scripts/greenfield/seal_short_decoder_ws32.py").read_text(encoding="utf-8")
    record = root / "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    pinned = re.search(r"^readonly DSA_ADJUDICATION_RECORD_8K_SHA=([0-9a-f]{64})$", wrapper, re.M)
    assert pinned is not None
    assert pinned.group(1) == hashlib.sha256(record.read_bytes()).hexdigest()
    assert "GLM_GREENFIELD_WS32_DSA_ADJUDICATION:-0" in wrapper
    assert "GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_ACK:-0" in wrapper
    assert wrapper.count("$DSA_ADJUDICATION_CLI") >= 2
    assert "--later-event-alarm-acknowledged" in sealer
    assert "requires an acknowledged lessons entry before sealing" in sealer
    assert "bind_ws32_adjudication(" in sealer


def test_ws32_shm_transport_is_pinned_to_the_sealed_checkpoint_identity() -> None:
    import hashlib
    import re
    import subprocess

    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    shm_pack = (root / "scripts/greenfield/run_ws32_runtime_checkpoint_shm_pack.sh").read_text(encoding="utf-8")
    cleanup = (root / "scripts/greenfield/cleanup_ws32_runtime_checkpoint_shm.sh").read_text(encoding="utf-8")
    for script in ("run_short_decoder_ws32.sh", "run_ws32_runtime_checkpoint_shm_pack.sh", "cleanup_ws32_runtime_checkpoint_shm.sh"):
        subprocess.run(["/usr/bin/bash", "-n", str(root / "scripts/greenfield" / script)], check=True)

    def pinned(source: str, name: str) -> str:
        match = re.search(rf"^readonly {name}=([0-9a-f]{{64}})$", source, re.M)
        assert match is not None, name
        return match.group(1)

    sealed_tag = "greenfield_ws32_runtime_pack_20260815T214050854386790Z"
    assert "GLM_GREENFIELD_WS32_CHECKPOINT_TRANSPORT:-gcsfuse" in wrapper
    assert f"readonly SHM_CHECKPOINT_ROOT=/dev/shm/glm-ws32-runtime/{sealed_tag}" in wrapper
    assert f"readonly SHM_ROOT=/dev/shm/glm-ws32-runtime/$SEALED_TAG" in shm_pack
    assert f"readonly SEALED_TAG={sealed_tag}" in shm_pack and f"readonly SEALED_TAG={sealed_tag}" in cleanup
    manifest_sha = pinned(wrapper, "SHM_MANIFEST_FILE_SHA")
    success_sha = pinned(wrapper, "SHM_SUCCESS_FILE_SHA")
    assert pinned(shm_pack, "SEALED_MANIFEST_FILE_SHA") == manifest_sha
    assert pinned(shm_pack, "SEALED_SUCCESS_FILE_SHA") == success_sha
    assert pinned(shm_pack, "SEALED_MANIFEST_SELF_SHA") == "c04f800edf15651198ab2c5183fff8a8ae9a427609b59cc61ccabdab32f5ee08"
    assert pinned(shm_pack, "SEALED_SUCCESS_SELF_SHA") == "1bfea5bd2dd8b096a3e551d7f96697496d8277bdaaa328ca1c35144feb8f1760"
    lineage = Path(f"/home/gianl/gcs-models/results/{sealed_tag}")
    assert hashlib.sha256((lineage / "checkpoint_manifest/manifest.json").read_bytes()).hexdigest() == manifest_sha
    assert hashlib.sha256((lineage / "checkpoint_SUCCESS.json").read_bytes()).hexdigest() == success_sha
    # Host sync in shm mode requires tmpfs, the pinned root, both sealed byte identities and four slots;
    # gcsfuse mode keeps the original mount check.
    sync = wrapper[wrapper.index("sync_command='") : wrapper.index("sync_rc=0")]
    assert 'findmnt -T "$checkpoint" -n -o FSTYPE | grep -qx tmpfs' in sync
    assert '[[ $checkpoint == "$shm_root" ]]' in sync
    assert '$(ls "$checkpoint"/device_slot_*.safetensors | wc -l) -eq 4' in sync
    assert 'grep -q "driftbench-dsv4-uc fuse.gcsfuse"; fi; echo "SYNC_OK' in sync
    # The shm pack proves every slot byte-identical to the sealed manifest and never uploads slots.
    assert "is NOT byte-identical to the sealed checkpoint" in shm_pack
    assert "slots_identical_to_sealed" in shm_pack
    assert "gcloud storage cp" in shm_pack and "device_slot_*.safetensors" in shm_pack
    host = shm_pack[shm_pack.index("host_command='") : shm_pack.index("pack_rc=0")]
    assert "gcloud storage cp --no-clobber \"$file\"" not in host
    assert "GLM_GREENFIELD_WS32_SHM_PACK:-0" in shm_pack and "GLM_GREENFIELD_WS32_SHM_CLEANUP:-0" in cleanup
    assert "/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_*" in cleanup
    # T1/T2: local-slot verification plumbed through runner and sealer; ERR trap and identity marker.
    runner = (root / "scripts/greenfield/run_short_decoder_ws32.py").read_text(encoding="utf-8")
    sealer = (root / "scripts/greenfield/seal_short_decoder_ws32.py").read_text(encoding="utf-8")
    assert '"--checkpoint-transport", choices=("gcsfuse", "shm")' in runner
    assert 'local_slot_layout=args.checkpoint_transport == "shm"' in runner
    assert '"checkpoint_transport": args.checkpoint_transport' in runner
    assert '"checkpoint_transport",' in sealer and 'record.get("checkpoint_transport") != args.checkpoint_transport' in sealer
    assert "--checkpoint-transport '\"$CHECKPOINT_TRANSPORT\"'" in wrapper
    assert 'marker="$checkpoint.identity.json"; [[ -f $marker ]]' in sync
    assert "trap fail ERR" in host and 'marker=Path(str(root)+".identity.json")' in host
    assert '[[ -f $shm.identity.json ]] || fail' in host
    assert 'chmod 0444 "$shm/manifest.json" "$shm/SUCCESS"' in host


def test_ws32_shell_wrappers_never_reference_a_local_in_its_own_declaration() -> None:
    """`local a=$1 b=${a}` is undefined under `set -u` (burned a tag on 2026-09-05 and 2026-08-16)."""
    import re

    root = Path(__file__).resolve().parents[3]
    for script in (
        "run_short_decoder_ws32.sh",
        "run_ws32_runtime_checkpoint_shm_pack.sh",
        "cleanup_ws32_runtime_checkpoint_shm.sh",
        "run_ws32_runtime_checkpoint_pack.sh",
    ):
        for line in (root / "scripts/greenfield" / script).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("local "):
                continue
            names = re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=", stripped[len("local "):])
            for name in names:
                assert not re.search(rf"\$\{{?{name}\b", stripped.split(f"{name}=", 1)[1] if f"{name}=" in stripped else ""), (
                    script,
                    line,
                )


def test_ws32_sealer_binds_alarm_acknowledgement_and_states_adjudicated_basis() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    sealer_path = root / "scripts/greenfield/seal_short_decoder_ws32.py"
    sealer = sealer_path.read_text(encoding="utf-8")
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    for token in (
        '"--later-event-alarm-profile"',
        '"--later-event-alarm-lessons-pin"',
        '"--recovery-code-hash"',
        "alarm acknowledgement is not bound to a profile record and lessons pin",
        "lacks a GATE_D_LESSONS entry naming this run",
        '"dsa_adjudication": expected_dsa_adjudication,',
        '"checkpoint_transport": args.checkpoint_transport,',
        'summary["later_event_alarm"] = alarm_summary',
        "DSA_EVENT0_EXACT",
        "LATER_EVENTS_RECORDED_NOT_ADJUDICATED",
        "DEEP_LAYER_TENSORS_NOT_BOUNDED_IN_THIS_RUN",
        "gate_d_s21_exact_tokens_adjudicated_dsa_state_cache",
    ):
        assert token in sealer, token
    assert "GLM_GREENFIELD_WS32_LATER_EVENT_ALARM_PROFILE_SHA" in wrapper
    assert '--recovery-code-hash "$RECOVERY_PIN"' in wrapper
    assert "${LATER_EVENT_ALARM_CLI:+$LATER_EVENT_ALARM_CLI}" in wrapper
    specification = importlib.util.spec_from_file_location("ws32_sealer_for_test", sealer_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    steps = [
        {"adjudication": {"alarm_events": [14, 17], "recorded_divergence_sizes": [
            {"event_index": 2, "symmetric_difference": 16}, {"event_index": 14, "symmetric_difference": 1030}]}},
        {"adjudication": {"alarm_events": [], "recorded_divergence_sizes": [
            {"event_index": 17, "symmetric_difference": 1948}]}},
    ]
    producers = [0, 1, 2] + list(range(6, 78, 4))
    summary = module._later_event_alarm_summary(steps, producers)
    assert summary["alarmed_step_count"] == 1 and summary["events_by_step"] == {"0": [14, 17]}
    assert summary["maximum_symmetric_difference"] == {
        "event_index": 17, "producer_layer_id": 62, "step": 1, "symmetric_difference": 1948}
    assert module._later_event_alarm_summary(
        [{"adjudication": {"alarm_events": [], "recorded_divergence_sizes": []}}], producers
    ) is None


def test_ws32_wrapper_recovery_archives_the_stale_source_ledger() -> None:
    """A failed post-materialization attempt leaves a ledger bound to the old
    recovery pin; recovery must move it aside instead of refusing forever."""
    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    start = wrapper.index("archive_failed_publication() {")
    body = wrapper[start : wrapper.index("\n}\n", start)]
    assert 'with_ledger == with_ledger && -e $RUN_DIR/source_remote_objects.json' in body
    # Only the recovery-start call site moves the ledger; on_exit keeps it as
    # rollback authority for a later attempt.
    assert wrapper.count("archive_failed_publication with_ledger") == 1
    assert wrapper.index("archive_failed_publication with_ledger\nfi") < wrapper.index(
        'say "materializing generation-pinned all-host evidence'
    )
    assert "a prior terminal SUCCESS verification exists; refusing recovery" in wrapper


def test_ws32_chunked_prefill_schema_is_consistent_across_runner_wrapper_sealer() -> None:
    """Spec §23.2: two prefill graphs (chunk + tail) everywhere, lint-checked as
    ``prefill``; the wrapper pins both and passes the chunk length and wall budget."""
    from glm_tpu.greenfield.validation import ws32_evidence

    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    sealer = (root / "scripts/greenfield/seal_short_decoder_ws32.py").read_text(encoding="utf-8")
    runner = (root / "scripts/greenfield/run_short_decoder_ws32.py").read_text(encoding="utf-8")
    assert ws32_evidence.BASE_GRAPHS == (
        "prefill_chunk", "prefill_tail", "observer", "decode", "cache_probe"
    )
    assert '"prefill",' not in runner and '"prefill",' not in sealer
    for graph in ("prefill_chunk", "prefill_tail"):
        assert RUNNER._linter_kind(graph) == "prefill"
        dashed = graph.replace("_", "-")
        assert f"--expected-{dashed}-stablehlo-sha256" in wrapper
        assert f"--expected-{dashed}-optimized-hlo-sha256" in wrapper
        assert f'"{dashed}",' in sealer and f'"{dashed}",' in runner
        assert f"{graph.upper()}_STABLE_SHA=$ZERO_SHA" in wrapper
    assert RUNNER._linter_kind("decode") == "decode"
    assert "for graph in exact_materialize exact_promote prefill_chunk prefill_tail observer decode cache_probe" in wrapper
    assert "--prefill-chunk '\"$PREFILL_CHUNK\"'" in wrapper
    assert "--prefill-budget-seconds '\"$PREFILL_BUDGET_SECONDS\"'" in wrapper
    assert "PREFILL_CHUNK=${GLM_GREENFIELD_WS32_PREFILL_CHUNK:-2048}" in wrapper
    assert '"prefill_chunk_length",' in sealer and '"prefill_execution",' in sealer
    assert "_require_prefill_execution(" in sealer
    assert "GLM_GREENFIELD_WS32_PREFILL_STABLEHLO_SHA" not in wrapper


def test_ws32_sealer_prefill_execution_accounting_is_fail_closed() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    sealer_path = root / "scripts/greenfield/seal_short_decoder_ws32.py"
    specification = importlib.util.spec_from_file_location("ws32_sealer_prefill_test", sealer_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    good = {
        "prefill_chunk_length": 2048,
        "prefill_execution": {
            "budget_seconds": 3600.0,
            "chunk_length": 2048,
            "chunk_wall_seconds": [270.5, 271.0, 269.9],
            "full_chunks": 3,
            "projected_total_seconds_max": 1090.0,
            "repaired_index_installed": True,
            "tail_length": 2011,
            "tail_wall_seconds": 266.0,
            "total_seconds": 1080.0,
        },
    }
    module._require_prefill_execution(good, mode="numerical", prompt_length=8155, rank=0)
    module._require_prefill_execution(
        {"prefill_chunk_length": 2048, "prefill_execution": None},
        mode="acquire",
        prompt_length=8155,
        rank=0,
    )
    with pytest.raises(SystemExit, match="recorded a prefill execution"):
        module._require_prefill_execution(good, mode="acquire", prompt_length=8155, rank=0)
    for mutate, message in (
        (lambda e: e.update(tail_length=2010), "accounting"),
        (lambda e: e.update(chunk_wall_seconds=[270.5, 271.0]), "accounting"),
        (lambda e: e.update(total_seconds=3601.0), "accounting"),
        (lambda e: e.update(projected_total_seconds_max=3600.5), "accounting"),
        (lambda e: e.update(total_seconds=100.0), "accounting"),
        (lambda e: e.pop("budget_seconds"), "schema"),
    ):
        record = {"prefill_chunk_length": 2048, "prefill_execution": dict(good["prefill_execution"])}
        mutate(record["prefill_execution"])
        with pytest.raises(SystemExit, match=message):
            module._require_prefill_execution(record, mode="numerical", prompt_length=8155, rank=0)
    with pytest.raises(SystemExit, match="chunk length"):
        module._require_prefill_execution(
            {"prefill_chunk_length": 0, "prefill_execution": None}, mode="acquire", prompt_length=8155, rank=0
        )


def test_ws32_sealer_binds_the_prefill_chunk_into_the_run_tag() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    sealer_path = root / "scripts/greenfield/seal_short_decoder_ws32.py"
    specification = importlib.util.spec_from_file_location("ws32_sealer_tag_test", sealer_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    stamp = "20260905T085534575653049Z"
    module._validate_run_tag(f"greenfield_ws32_short_decoder_8k_numerical_{stamp}", context_label="8k", mode="numerical")
    module._validate_run_tag(
        f"greenfield_ws32_short_decoder_8k_numerical_c512_{stamp}", context_label="8k", mode="numerical", prefill_chunk=512
    )
    for tag, chunk in (
        (f"greenfield_ws32_short_decoder_8k_numerical_{stamp}", 512),
        (f"greenfield_ws32_short_decoder_8k_numerical_c512_{stamp}", 2048),
        (f"greenfield_ws32_short_decoder_8k_numerical_c2048_{stamp}", 2048),
        (f"greenfield_ws32_short_decoder_8k_acquire_c512_{stamp}", 512),
    ):
        with pytest.raises(SystemExit, match="prefill chunk"):
            module._validate_run_tag(tag, context_label="8k", mode="numerical", prefill_chunk=chunk)
    with pytest.raises(SystemExit, match="chunk length drifted"):
        module._require_prefill_execution(
            {"prefill_chunk_length": 2048, "prefill_execution": None},
            mode="acquire", prompt_length=8155, rank=0, expected_chunk=512,
        )
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    assert 'CHUNK_SUFFIX=_c${PREFILL_CHUNK}' in wrapper
    assert '${MODE}${CHUNK_SUFFIX}_[0-9]{8}T[0-9]{15}Z$' in wrapper
    assert '--prefill-chunk "$PREFILL_CHUNK"' in wrapper


def test_ws32_capacity_and_rotary_diagnostic_are_declared_and_bound() -> None:
    """Spec §23.3 Step C / §23.8: capacity enters the tag and DB identity; the
    rotary diagnostic is a declared side program whose record must be present
    iff enabled, self-consistent, script-pinned and identical across ranks."""
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    sealer_path = root / "scripts/greenfield/seal_short_decoder_ws32.py"
    specification = importlib.util.spec_from_file_location("ws32_sealer_cap_test", sealer_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    stamp = "20260905T085534575653049Z"
    module._validate_run_tag(
        f"greenfield_ws32_short_decoder_8k_numerical_cap131072_{stamp}",
        context_label="8k", mode="numerical", context_capacity=131072,
    )
    module._validate_run_tag(
        f"greenfield_ws32_short_decoder_8k_numerical_c512_cap262656_{stamp}",
        context_label="8k", mode="numerical", prefill_chunk=512, context_capacity=262656,
    )
    for tag, chunk, capacity in (
        (f"greenfield_ws32_short_decoder_8k_numerical_{stamp}", 2048, 131072),
        (f"greenfield_ws32_short_decoder_8k_numerical_cap131072_{stamp}", 2048, 8192),
        (f"greenfield_ws32_short_decoder_8k_numerical_cap131072_c512_{stamp}", 512, 131072),
    ):
        with pytest.raises(SystemExit, match="capacity"):
            module._validate_run_tag(tag, context_label="8k", mode="numerical", prefill_chunk=chunk, context_capacity=capacity)
    # DB identity for a capacity measurement is never a Gate D row.
    base = {"dsa_adjudication": {"event_index": 1}, "later_event_alarm": {"acknowledged": True}}
    note, item_id, gold = module._run_rows(base)
    assert item_id == "gate_d_s21_exact_tokens_adjudicated_dsa_state_cache"
    note_c, item_c, gold_c = module._run_rows({**base, "capacity_measurement": {"context_capacity": 131072, "default": 8192}})
    assert item_c == "s23_capacity_measurement_cap131072_gate_d_s21_exact_tokens_adjudicated_dsa_state_cache"
    assert note_c.startswith("Capacity measurement at context_capacity=131072") and "not a Gate D record" in note_c
    assert gold_c == gold
    env = module._run_environment({**base, "checkpoint_manifest_sha256": "1", "checkpoint_success_sha256": "2", "code_hash": "3",
        "context_label": "8k", "dsa_oracle_manifest_sha256": "4", "dsa_oracle_success_sha256": "5", "mesh_sha256": "6",
        "run_tag": "t", "strategy_nd_dense": False, "strategy_nd_dense_overlay_manifest_sha256": "0" * 64,
        "token_oracle_manifest_sha256": "7", "token_oracle_success_sha256": "8", "xla_python_client_mem_fraction": ".95",
        "capacity_measurement": {"context_capacity": 131072}, "rotary_diagnostic": {"verdict": "PASS"}})
    assert env["capacity_measurement"] == {"context_capacity": 131072} and env["rotary_diagnostic"] == {"verdict": "PASS"}
    # Rotary diagnostic presence must match the declaration.
    from glm_tpu.greenfield.validation import rotary_diagnostic as rd
    record = rd.run_rotary_diagnostic(positions=600).record
    # A CPU-computed or shortened record can never seal (protocol pinned, TPU only).
    with pytest.raises(SystemExit, match="invalid.*(pre-registered|expected 'tpu')"):
        module._require_rotary_diagnostic({"rotary_diagnostic": record}, enabled=True, rank=0, first={"rotary_diagnostic": record})
    with pytest.raises(SystemExit, match="present but not declared"):
        module._require_rotary_diagnostic({"rotary_diagnostic": record}, enabled=False, rank=0, first={})
    with pytest.raises(SystemExit, match="invalid"):
        module._require_rotary_diagnostic({"rotary_diagnostic": None}, enabled=True, rank=0, first={})
    module._require_rotary_diagnostic({"rotary_diagnostic": None}, enabled=False, rank=3, first={})
    assert "expected_backend=\"tpu\", pinned=True" in sealer_path.read_text(encoding="utf-8")
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    runner = (root / "scripts/greenfield/run_short_decoder_ws32.py").read_text(encoding="utf-8")
    assert "CONTEXT_CAPACITY=${GLM_GREENFIELD_WS32_CONTEXT_CAPACITY:-8192}" in wrapper
    assert "CHUNK_SUFFIX=${CHUNK_SUFFIX}_cap${CONTEXT_CAPACITY}" in wrapper
    assert "ROTARY_DIAGNOSTIC=${GLM_GREENFIELD_WS32_ROTARY_DIAGNOSTIC:-0}" in wrapper
    assert "--rotary-diagnostic '\"$ROTARY_DIAGNOSTIC\"'" in wrapper and '--rotary-diagnostic "$ROTARY_DIAGNOSTIC"' in wrapper
    assert '"--rotary-diagnostic", choices=(0, 1), default=0' in runner
    assert '"rotary_diagnostic": rotary_diagnostic,' in runner
    sealer = sealer_path.read_text(encoding="utf-8")
    assert '"rotary_diagnostic",' in sealer and "ROTARY_LONG_POSITION_DIAGNOSTIC_" in sealer and "CAPACITY_MEASUREMENT_" in sealer


def test_ws32_evidence_layout_v2_is_declared_end_to_end() -> None:
    """Storage plan 2026-09-05 (Proposal A): one gzip per graph/form uploaded
    redundantly by every rank with --no-clobber and CRC-matched tolerance; the
    record declares the layout; the wrapper refuses launches without storage
    headroom."""
    from glm_tpu.greenfield.validation import ws32_evidence

    root = Path(__file__).resolve().parents[3]
    wrapper = (root / "scripts/greenfield/run_short_decoder_ws32.sh").read_text(encoding="utf-8")
    runner = (root / "scripts/greenfield/run_short_decoder_ws32.py").read_text(encoding="utf-8")
    sealer = (root / "scripts/greenfield/seal_short_decoder_ws32.py").read_text(encoding="utf-8")
    assert '"evidence_layout": EVIDENCE_LAYOUT_V2,' in runner and '"evidence_layout": EVIDENCE_LAYOUT_V2,' in sealer
    assert 'record.get("evidence_layout") != EVIDENCE_LAYOUT_V2' in sealer and '"evidence_layout",' in sealer
    assert "upload_shared(){" in wrapper and 'gzip -n -9 -c "$hlo/$graph.$form"' in wrapper
    assert '"$remote/hlo/${graph}.${form}.gz"' in wrapper and ".rank${idx}.stablehlo.mlir" not in wrapper
    assert "STORAGE_CEILING_BYTES=2500000000000" in wrapper and "STORAGE_RESERVE_BYTES=6000000000" in wrapper
    assert "live storage plus reserve exceeds the ceiling" in wrapper
    assert ws32_evidence.EVIDENCE_LAYOUT_V2 == "hlo_single_gzip_v2"

