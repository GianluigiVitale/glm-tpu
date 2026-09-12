"""Full long sealer control path, fixture HLO/math/trace, real memory/accounting.

Preparation originals and compiler replay are tested separately. These fixtures
prove orchestration and classifications only, never a model or hardware result.
"""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.greenfield import seal_short_decoder_ws32 as sealer
from scripts.greenfield import ws32_delivery_runtime as runtime
from scripts.greenfield import ws32_delivery_phase_evidence as phases
from tests.greenfield.validation.test_ws32_batched_composed_sealer import build, ROOT
from tests.greenfield.validation.test_ws32_delivery_companions import request
from tests.greenfield.validation.test_ws32_delivery_runtime import fleet
from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES


@pytest.mark.parametrize("label,reused_tail", [
    ("128k_d1_0", False), ("256k_e0", False), ("256k_e0", True),
])
def test_actual_long_sealer_to_summary_and_db_contract(tmp_path, monkeypatch, label, reused_tail):
    args, _, memories = build(tmp_path, monkeypatch)
    args.__dict__.update(request(label).__dict__)
    plan = runtime.programs.long_plan(label)
    args.context_label = label
    args.tag = (f"greenfield_ws32_short_decoder_{label}_numerical_c128_cap{plan.context_capacity}"
                "_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T070000000000000Z")
    args.long_context_oracle_dir = tmp_path / "long_oracle"
    args.tokenizer_root = tmp_path / "tokenizer"
    args.token_oracle_dir = args.dsa_oracle_dir = None
    for name in ("token_oracle_manifest", "token_oracle_success", "dsa_oracle_manifest", "dsa_oracle_success"):
        setattr(args, name + "_sha256", "0" * 64)
    entry = WS32_LONG_CONTEXT_PROFILES[label]
    oracle = NS(**{k: entry[k] for k in ("kind", "depth", "item_row_id", "source_run_id", "manifest_sha256")},
                generated_token_ids=np.zeros(entry["generated_token_count"], np.int32),
                prompt_token_ids=np.zeros(plan.prompt_length, np.int32))
    monkeypatch.setattr(sealer, "load_ws32_long_context_oracle", lambda *a, **k: oracle)
    comparison = dict(passkey_matches_gold=True if entry["kind"] == "passkey" else None)
    monkeypatch.setattr(sealer, "_long_context_token_result", lambda *a, **k: comparison)
    monkeypatch.setattr(sealer, "compare_ws32_dsa_within_engine", lambda **k: dict(passed=True))
    # Actual rotary table validation has separate full-capacity tests; this
    # branch test does not allocate eight identical tables for a schema check.
    monkeypatch.setattr(sealer, "_require_main_rope_table", lambda *a, **k: None)
    reports, replays, phase_calls = {}, [], []
    def replay(stable, optimized, *, graph, args, **kwargs):
        source = "prefill_chunk" if reused_tail and graph == "prefill_tail" else graph
        assert stable == f"fixture stable {source}" and optimized == f"fixture optimized {source}"
        if graph.startswith("wk_"): assert kwargs["compiled_memory"] == wk_memory
        replays.append(graph)
        return reports[graph]
    monkeypatch.setattr(sealer, "_replay_batched_graph", replay)
    def phase_check(**kwargs):
        assert kwargs["root"] == tmp_path / "fleet" and len(kwargs["records"]) == 8
        assert len(kwargs["full_index_layers"]) == 21
        assert all(r["delivery_decode_preparation"] == {"fixture": True} for r in kwargs["records"])
        phase_calls.append(True)
        return dict(fixture_only=True)
    monkeypatch.setattr(phases, "validate_fleet", phase_check)
    f = fleet(label)
    wk_memory = dict(alias_size_in_bytes=0, argument_size_in_bytes=1024,
                     generated_code_size_in_bytes=1024, output_size_in_bytes=1024, temp_size_in_bytes=1024)
    for graph in (*runtime.ROLES, "observer", "decode", "cache_probe", "exact_materialize", "exact_promote", "wk_decode", "wk_promote"):
        source = "prefill_chunk" if reused_tail and graph == "prefill_tail" else graph
        stable, optimized = f"fixture stable {source}", f"fixture optimized {source}"
        reports[graph] = dict(stablehlo_sha256=sha256(stable.encode()).hexdigest(),
                              optimized_hlo_sha256=sha256(optimized.encode()).hexdigest())
        for rank in range(8):
            (tmp_path / f"fleet_hlo/{graph}.rank{rank}.stablehlo.mlir").write_text(stable)
            (tmp_path / f"fleet_hlo/{graph}.rank{rank}.optimized_hlo.txt").write_text(optimized)
    for rank in range(8):
        path = tmp_path / f"fleet/runner.rank{rank}.json"
        record = json.loads(path.read_text())
        record.update(runtime.numerical_identity(label))
        record.update(prompt_length=plan.prompt_length, prefill_chunk_length=128,
            long_context={k: getattr(oracle, k) for k in ("depth", "item_row_id", "kind", "manifest_sha256", "source_run_id")}
                         | dict(success_sha256=args.long_context_success_sha256),
            graphs=reports, delivery_wk_phase={"fixture": True},
            delivery_decode_preparation={"fixture": True}, delivery_phase_timings={"fixture": True})
        for name in ("token_oracle_manifest_sha256", "token_oracle_success_sha256", "dsa_oracle_manifest_sha256", "dsa_oracle_success_sha256"):
            record[name] = None
        n = 1 + args.observer_steps + args.warmup + args.iterations + args.trace_steps
        record.update(observed_generated_token_ids=[100] * n, token_comparison=comparison,
                      dsa_steps=[dict(passed=True)] * args.observer_steps,
                      state=dict(position=[plan.prompt_length + n - 1],
                                 context_lengths=[plan.prompt_length + n], contract_valid=[True]))
        record["compile_seconds"].update(dict.fromkeys(("wk_decode", "wk_promote"), 1.0))
        record["compiled_memory_analysis"].update(f["records"][rank]["compiled_memory_analysis"])
        record["compiled_memory_analysis"].update({name: wk_memory for name in ("wk_decode", "wk_promote")})
        if reused_tail:
            record["compile_seconds"]["prefill_tail"] = 0.0
            record["compiled_memory_analysis"]["prefill_tail"] = deepcopy(
                record["compiled_memory_analysis"]["prefill_chunk"])
            # Exercise phase-sensitive allocator stats in the actual sealer,
            # not only the standalone predicate. HBM/owner joins remain below.
            for field in ("device_memory_after_compile", "device_memory_after_execute"):
                for memory in record[field]:
                    memory.update(bytes_reserved=0, peak_bytes_reserved=200,
                                  bytes_reservable_limit=100)
        record["batched_prefill_memory"] = f["records"][rank]["batched_prefill_memory"]
        record["batched_device_memory_after_execute"] = f["records"][rank]["batched_device_memory_after_execute"]
        record["prefill_execution"].update(identity=runtime.identity(label),
            budget_seconds=runtime.budget_seconds(label), final_frontier=plan.prompt_length,
            input_transfer_seconds=[.001] * (plan.split[0] + 1),
            block_wall_seconds=[.1] * (plan.split[0] + 1), request_prefill_seconds=1000.,
            projected_total_seconds_max=1000.,
            memory_admission=f["records"][rank]["prefill_execution"]["memory_admission"])
        timing = record["profiler_free_timing"]
        timing.update(iterations=args.iterations, samples_ms=[100.] * args.iterations,
                      distribution=sealer._distribution([100.] * args.iterations))
        npz = path.with_suffix('.npz')
        with np.load(npz) as loaded: arrays = {k: loaded[k] for k in loaded.files}
        for name in ("dsa_selected_positions", "dsa_selected_scores", "dsa_selected_valid_counts"):
            arrays[name] = np.zeros((args.observer_steps, 1), arrays[name].dtype)
        np.savez(npz, **arrays)
        record["numerical_tensors"] = dict(filename=npz.name, byte_count=npz.stat().st_size,
            sha256=sealer._digest_file(npz), arrays={name: dict(dtype=v.dtype.name, shape=list(v.shape),
                sha256=sha256(v.tobytes()).hexdigest()) for name, v in arrays.items()})
        path.write_text(json.dumps(record))
    assert sealer._validate(args) == 0
    result = json.loads(args.output.read_text())
    assert len(replays) == 9 and len(memories) == 1 and memories[0]["owner_count"] == 32
    assert phase_calls == [True] and "BATCHED_PREFILL_LONG_S26" in result["classification"]
    assert result["verified_generated_token_count"] is None
    assert sealer._run_rows(result)[1].startswith("s26_batched_s23_5_long_context_")
    if label == "256k_e0":
        assert "NO_CORRECTNESS_ORACLE" in result["classification"]
        assert sealer._run_rows(result)[3:] == (None, None)
