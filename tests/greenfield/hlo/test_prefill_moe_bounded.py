"""New bounded protocol must not relax old exact admission or trust verdicts."""

import copy
from hashlib import sha256

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield.prefill_moe_numerical import compare_outputs
from scripts.greenfield import probe_ws32_prefill_moe as probe
from scripts.greenfield import ws32_prefill_moe_campaign as campaign
from tests.greenfield.hlo.test_prefill_real_moe_admission import (
    valid_workers,
    valid_hlo,
)


def fp32_hlo():
    add = """%add {
%p0 = f32[] parameter(0)
%p1 = f32[] parameter(1)
ROOT %out = f32[] add(%p0, %p1)
}
%sum_fusion {
%routes = f32[8,17,1536] parameter(0)
%zero = f32[] constant(0)
ROOT %sum = f32[17,1536] reduce(%routes, %zero), dimensions={0}, to_apply=%add, metadata={op_name="greenfield_ws32_prefill_moe/fp32_route_sum/reduce_sum"}
}
"""
    hlo = valid_hlo().replace(
        "ENTRY %main {",
        add
        + """ENTRY %main {
%routes_in = f32[8,17,1536] parameter(0)
%sum_call = f32[17,1536] fusion(%routes_in), calls=%sum_fusion
""",
    )
    return hlo.replace(
        "%e = bf16[17,1536] all-reduce(%f)", "%e = bf16[17,1536] all-reduce(%sum_call)"
    ).replace(
        "replica_groups={{0,4,8,12,16,20,24,28}",
        "to_apply=%add, replica_groups={{0,4,8,12,16,20,24,28}",
    )


def test_per_row_caps_and_direct_legacy_comparison_are_required():
    a = np.ones((17, 1536), dtype=ml_dtypes.bfloat16)
    b = a.copy()
    legacy = a[:1].copy()
    a[2, 0] = 1.0625
    assert compare_outputs(a, b, legacy)["passed"]
    # Aggregate mean remains small, but one row's mean/p99 violates bounds.
    a[2, :] = 1.125
    result = compare_outputs(a, b, legacy)
    assert not result["passed"] and not result["per_row"][2]["passed"]
    a = b.copy()
    legacy[:] = 1.25
    assert not compare_outputs(a, b, legacy)["passed"]
    a[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compare_outputs(a, b, legacy)


def test_fp32_hlo_contract_rejects_missing_or_bf16_sum():
    assert probe.check_hlo(fp32_hlo(), fp32_route_sum=True)["passed"]
    assert not probe.check_hlo(valid_hlo(), fp32_route_sum=True)["passed"]
    assert not probe.check_hlo(
        fp32_hlo().replace("%sum = f32", "%sum = bf16"), fp32_route_sum=True
    )["passed"]
    for bad in (
        fp32_hlo().replace("all-reduce(%sum_call)", "all-reduce(%f)"),
        fp32_hlo()
        .replace(
            "%e = bf16",
            "%rounded = bf16[17,1536] convert(%sum_call)\n%promoted = f32[17,1536] convert(%rounded)\n%e = bf16",
        )
        .replace("all-reduce(%sum_call)", "all-reduce(%promoted)"),
        fp32_hlo().replace(
            "ROOT %out = f32[] add(%p0, %p1)",
            "ROOT %out = f32[] add(%p0, %p1), float_type_correction_info={original_type=BF16}",
        ),
    ):
        assert not probe.check_hlo(bad, fp32_route_sum=True)["passed"]


def test_fp32_sum_forwards_through_fusion_parameter():
    hlo = (
        fp32_hlo()
        .replace(
            "ENTRY %main {",
            """%forward {
%arg = f32[17,1536] parameter(0)
ROOT %copy = f32[17,1536] copy(%arg)
}
ENTRY %main {""",
        )
        .replace(
            "%e = bf16",
            "%forwarded = f32[17,1536] fusion(%sum_call), calls=%forward\n%e = bf16",
        )
        .replace("all-reduce(%sum_call)", "all-reduce(%forwarded)")
    )
    assert probe.check_hlo(hlo, fp32_route_sum=True)["passed"]
    assert not probe.check_hlo(
        fp32_hlo().replace(
            "dimensions={0}",
            "float_type_correction_info={original_type=BF16}, dimensions={0}",
        ),
        fp32_route_sum=True,
    )["passed"]


def test_bounded_mode_is_distinct_and_replays_original_outputs(tmp_path):
    records = valid_workers()
    for r in records:
        r.update(
            protocol=probe.BOUNDED_PROTOCOL, bounded_admission=True, fp32_route_sum=True
        )
        for case in probe.CASES:
            for s in r["cases"][case]["shards"]:
                s.update(bit_mismatches=1, bounded_comparison={"passed": True})
    campaign.validate_workers(records, "b" * 40, bounded=True)
    with pytest.raises(ValueError):
        campaign.validate_workers(records, "b" * 40)
    changed = copy.deepcopy(records)
    changed[0]["fp32_route_sum"] = False
    with pytest.raises(ValueError):
        campaign.validate_workers(changed, "b" * 40, bounded=True)
    hlo = fp32_hlo()
    (tmp_path / "candidate.optimized_hlo.txt").write_text(hlo)
    (tmp_path / "reference.optimized_hlo.txt").write_text("reference")
    record = records[0]
    record["hlo"]["sha256"] = sha256(hlo.encode()).hexdigest()
    record["reference_hlo_sha256"] = sha256(b"reference").hexdigest()
    actual = np.ones((17, 1536), dtype=ml_dtypes.bfloat16)
    reference = actual.copy()
    actual[1, 0] = 1.0625
    legacy = {c: np.ones((1, 6144), dtype=ml_dtypes.bfloat16) for c in probe.CASES}
    for case in probe.CASES:
        arrays = {}
        for s in record["cases"][case]["shards"]:
            arrays[f"actual_{s['device_id']}"] = actual.view(np.uint16)
            arrays[f"reference_{s['device_id']}"] = reference.view(np.uint16)
            s.update(
                output_sha256=sha256(actual.tobytes()).hexdigest(),
                reference_sha256=sha256(reference.tobytes()).hexdigest(),
                bounded_comparison=compare_outputs(
                    actual, reference, legacy[case][:, :1536]
                ),
            )
        np.savez_compressed(tmp_path / f"{case}.npz", **arrays)
    campaign.validate_files(tmp_path, record, bounded=True, legacy_outputs=legacy)
    record["cases"]["normal"]["shards"][0]["bounded_comparison"]["per_row"][1]["error"][
        "max_abs"
    ] = 0
    with pytest.raises(ValueError, match="bounded comparison"):
        campaign.validate_files(tmp_path, record, bounded=True, legacy_outputs=legacy)
