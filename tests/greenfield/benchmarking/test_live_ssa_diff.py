from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.live_ssa_diff import (
    ACCEPTED_LAYER0_HLO_SHA256,
    CANDIDATE_LAYER0_HLO_SHA256,
    _classify_producer,
    compare_layer0_live_ssa,
    compare_layer0_live_ssa_files,
    write_layer0_live_ssa_report,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


REAL_ACCEPTED_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_ACCEPTED_LAYER0_HLO",
        "/home/gianl/gcs-models/oracles/greenfield/glm52/"
        "decode_projection_lowering/8k/"
        "greenfield_accepted_decode_projection_lowering_"
        "20260811T184908676873350Z/accepted_decode_projection_lowering/"
        "jit_step_fun_impl.m32.after_codegen_hlo.txt.gz",
    )
)
REAL_CANDIDATE_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_ACCEPTED_SOURCE_LAYER0_HLO",
        "/home/gianl/glm-run/"
        "greenfield_strategy_nd_integrated_dense_accepted_source_context_"
        "20260814T213945734760843Z/hlo/"
        "strategy_nd_integrated_dense_accepted_source_context_"
        "bfloat16_32x6144.optimized_hlo.txt",
    )
)


def test_producer_classification_and_atomic_report(tmp_path: Path) -> None:
    module = parse_hlo_module(
        '''HloModule test, is_scheduled=true, num_partitions=32

ENTRY main (p: bf16[32,6144]) -> bf16[32,6144] {
  %p = bf16[32,6144]{1,0} parameter(0)
  ROOT %accepted = bf16[32,6144]{1,0} copy(%p), metadata={op_name="jit(step_fun_impl)/aten::embedding/jit(_take)/gather"}
}
'''
    )
    accepted = next(
        item for item in module.instructions if item.name == "%accepted"
    )
    assert _classify_producer(accepted) == "accepted_embedding_lookup"

    output = tmp_path / "report.json"
    write_layer0_live_ssa_report(output, {"status": "DIFF_IDENTIFIED"})
    assert json.loads(output.read_text()) == {"status": "DIFF_IDENTIFIED"}
    assert not (tmp_path / ".report.json.tmp").exists()


@pytest.mark.skipif(
    not (REAL_ACCEPTED_HLO.is_file() and REAL_CANDIDATE_HLO.is_file()),
    reason="SHA-pinned accepted/candidate TPU HLO pair absent",
)
def test_real_layer0_live_ssa_diff() -> None:
    report = compare_layer0_live_ssa_files(
        REAL_ACCEPTED_HLO,
        REAL_CANDIDATE_HLO,
        accepted_sha256=ACCEPTED_LAYER0_HLO_SHA256,
        candidate_sha256=CANDIDATE_LAYER0_HLO_SHA256,
    )
    assert report["status"] == "DIFF_IDENTIFIED"
    assert report["first_divergence"]["boundary"] == (
        "embedding_collective_input"
    )
    assert report["first_divergence"]["kind"] == "source_producer"
    assert report["first_divergence"]["accepted"]["mode"] == (
        "accepted_embedding_lookup"
    )
    assert report["first_divergence"]["candidate"]["mode"] == (
        "external_m1_embedding_reconstruction"
    )
    assert all(
        record["algorithm_result_match"] is True
        for record in report["collectives"].values()
    )
    assert all(
        record["semantic_match"] is True
        and record["backend_match"] is True
        for record in report["semantic_reductions"].values()
    )
    assert report["semantic_reductions"]["predense_reduction"][
        "physical_match"
    ] is False
    assert report["semantic_reductions"]["layer1_reduction"][
        "physical_match"
    ] is True
    assert all(
        record["match"] is True
        for record in report["scheduled_geometry"].values()
    )
    assert {
        label: record["fusion_count"]
        for label, record in report["layer1_output_paths"]["accepted"].items()
    } == {"attention": 1, "dense": 1, "embedding": 1}
    assert {
        label: record["fusion_count"]
        for label, record in report["layer1_output_paths"]["candidate"].items()
    } == {"attention": 2, "dense": 1, "embedding": 2}
    assert [
        (item["boundary"], item["kind"])
        for item in report["differences"]
    ] == [
        ("embedding_collective_input", "source_producer"),
        ("embedding_collective", "barrier_config"),
        ("attention_collective_input", "source_producer"),
        ("attention_collective", "barrier_config"),
        ("dense_collective_input", "source_producer"),
        ("dense_collective", "barrier_config"),
        ("predense_reduction", "fusion_boundary_layout"),
        ("layer1_output", "fusion_ownership"),
        ("layer1_output", "fusion_ownership"),
    ]

    with gzip.open(REAL_ACCEPTED_HLO, "rt", errors="replace") as stream:
        accepted_hlo = stream.read()
    candidate_hlo = REAL_CANDIDATE_HLO.read_text()
    swapped = candidate_hlo.replace(
        "fusion(%psum.23, %psum.22, %psum.21, %copy-done.5), ",
        "fusion(%psum.22, %psum.23, %psum.21, %copy-done.5), ",
        1,
    )
    assert swapped != candidate_hlo
    swapped_report = compare_layer0_live_ssa(accepted_hlo, swapped)
    assert swapped_report["semantic_reductions"]["layer1_reduction"][
        "semantic_match"
    ] is False

    dead_sources = candidate_hlo.replace(
        "%mul.109 = f32[1,6144]{1,0:T(1,128)} "
        "multiply(%convert.3, %convert.4)",
        "%mul.109 = f32[1,6144]{1,0:T(1,128)} "
        "multiply(%convert.4, %convert.4)",
        1,
    )
    assert dead_sources != candidate_hlo
    with pytest.raises(BenchmarkValidationError, match="no live SSA path"):
        compare_layer0_live_ssa(accepted_hlo, dead_sources)

    tuple_decoy = candidate_hlo.replace(
        "  %slice.42 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
        "slice(%param_0.118), slice={[0:1], [0:6144]}",
        "  %source_tuple = "
        "(bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}, "
        "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}) "
        "tuple(%param_0.118, %param_1.107)\n"
        "  %selected_source = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} "
        "get-tuple-element(%source_tuple), index=1\n"
        "  %slice.42 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
        "slice(%selected_source), slice={[0:1], [0:6144]}",
        1,
    )
    assert tuple_decoy != candidate_hlo
    parse_hlo_module(tuple_decoy)
    tuple_report = compare_layer0_live_ssa(accepted_hlo, tuple_decoy)
    assert tuple_report != report
    assert {
        label: tuple_report["layer1_output_paths"]["candidate"][label][
            "fusion_count"
        ]
        for label in ("attention", "embedding")
    } == {"attention": 3, "embedding": 3}
