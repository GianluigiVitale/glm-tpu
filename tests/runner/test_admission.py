"""Tests of :mod:`glm_tpu.runner.admission`: live-memory admission and the fresh HLO graph admission.

The boundary cases sit on the 512 MiB reserve and the 4 KiB full-pod payload (the verifier mutations that
once passed every gate: reserve -> 0, payload limit -> 1 MiB). The refusal messages, the report and its
profile string are pinned literally: only the frozen safety record's verdicts and the G1-/G2-protocol
reports record them otherwise, and one message (the partition/replica refusal) is in no record.
"""

from __future__ import annotations

import pytest

from glm_tpu.runner.admission import check_hlo_collectives, project_memory

GIB = 1 << 30
MIB = 1 << 20
LIMIT = 32 * GIB
IN_USE = 30 * GIB
RESERVE = 512 * MIB
FITS = LIMIT - IN_USE - RESERVE - 1  # the largest compiled footprint that fits: predicted = limit - 1


def chips(*in_use: int, ids: tuple[int, ...] = (0, 1, 2, 3)) -> list[dict]:
    return [
        dict(device_id=device, bytes_in_use=used, bytes_limit=LIMIT, peak_bytes_in_use=used)
        for device, used in zip(ids, in_use, strict=True)
    ]


def memory(output: object, temp: object = 0, code: object = 0, alias: object = 0) -> dict:
    return dict(
        output_size_in_bytes=output,
        temp_size_in_bytes=temp,
        generated_code_size_in_bytes=code,
        alias_size_in_bytes=alias,
    )


FOUR = chips(IN_USE, IN_USE, IN_USE, IN_USE)


# ------------------------------------------------------------------------------ project_memory
def test_one_byte_below_the_limit_fits_and_the_report_is_complete():
    report = project_memory(FOUR, memory(FITS - 3, 1, 2))
    assert report == dict(
        passed=True,
        reserve_bytes=RESERVE,
        chips=[dict(device_id=i, predicted_bytes=LIMIT - 1, limit=LIMIT, fits=True) for i in range(4)],
    )


def test_the_reserve_makes_exactly_the_limit_not_fit():
    report = project_memory(FOUR, memory(FITS + 1))
    assert not report["passed"]
    assert {row["predicted_bytes"] for row in report["chips"]} == {LIMIT}
    assert project_memory(FOUR, memory(FITS + 1), reserve_bytes=RESERVE - 1)["passed"]


def test_aliased_bytes_are_credited_but_never_below_zero():
    assert project_memory(FOUR, memory(FITS + GIB, alias=GIB))["passed"]
    assert not project_memory(FOUR, memory(FITS + GIB + 1, alias=GIB))["passed"]
    report = project_memory(FOUR, memory(0, alias=GIB))
    assert report["passed"] and {row["predicted_bytes"] for row in report["chips"]} == {IN_USE + RESERVE}


def test_every_chip_must_fit():
    report = project_memory(chips(IN_USE, IN_USE, IN_USE + 1, IN_USE), memory(FITS))
    assert not report["passed"]
    assert [row["fits"] for row in report["chips"]] == [True, True, False, True]


@pytest.mark.parametrize(
    ("stats", "value", "reserve", "message"),
    [
        (FOUR, memory(-1), RESERVE, "invalid compiler memory accounting"),
        (FOUR, memory(1.0), RESERVE, "invalid compiler memory accounting"),
        (FOUR, dict(output_size_in_bytes=1), RESERVE, "invalid compiler memory accounting"),
        (FOUR, memory(1), -1, "invalid compiler memory accounting"),
        (
            chips(IN_USE, IN_USE, IN_USE, ids=(0, 1, 2)),
            memory(1),
            RESERVE,
            "memory admission needs four distinct local chips",
        ),
        (
            chips(IN_USE, IN_USE, IN_USE, IN_USE, ids=(0, 1, 2, 2)),
            memory(1),
            RESERVE,
            "memory admission needs four distinct local chips",
        ),
        (chips(IN_USE, IN_USE, LIMIT + 1, IN_USE), memory(1), RESERVE, "invalid live allocator accounting"),
        (chips(IN_USE, IN_USE, -1, IN_USE), memory(1), RESERVE, "invalid live allocator accounting"),
    ],
    ids=[
        "negative",
        "float",
        "missing",
        "negative_reserve",
        "three_chips",
        "duplicate_chip",
        "over_limit",
        "negative_use",
    ],
)
def test_invalid_accounting_is_refused(stats, value, reserve, message):
    with pytest.raises(ValueError) as refused:
        project_memory(stats, value, reserve_bytes=reserve)
    assert str(refused.value) == message


# ------------------------------------------------------------------------------ check_hlo_collectives
FEATURE = "{" + ",".join("{" + ",".join(str(i) for i in range(r * 4, r * 4 + 4)) + "}" for r in range(8)) + "}"
EXPERT = "{" + ",".join("{" + ",".join(str(i) for i in range(c, 32, 4)) + "}" for c in range(4)) + "}"
POD = "{{" + ",".join(str(i) for i in range(32)) + "}}"
PAIRS = "{" + ",".join("{" + f"{i},{i + 1}" + "}" for i in range(0, 32, 2)) + "}"


def optimized_hlo(
    *,
    pod_elements: int = 1024,
    extra: str = "",
    feature_groups: str = FEATURE,
    collectives: bool = True,
    header: str = "num_partitions=32",
    dtype: str = "f32",
) -> str:
    """An optimized-HLO module in the TPU compiler's text form: an all-reduce over the feature axis, an
    all-gather over the expert axis and a full-pod all-reduce of ``pod_elements`` values (1,024 f32 values =
    the 4 KiB full-pod limit)."""
    tile, pod = f"{dtype}[8,128]{{1,0}}", f"{dtype}[{pod_elements}]{{0}}"
    reduce, gather = "use_global_device_ids=true, to_apply=%add", "dimensions={0}, use_global_device_ids=true"
    if collectives:
        body = (
            f"  %ar = {tile} all-reduce({tile} %c), channel_id=1, replica_groups={feature_groups}, {reduce}\n"
            f"  %ag = {dtype}[64,128]{{1,0}} all-gather({tile} %ar), channel_id=2, replica_groups={EXPERT}, {gather}\n"
            f"  %s = {pod} slice({dtype}[64,128]{{1,0}} %ag), slice={{[0:{pod_elements}]}}\n"
            f"  %pr = {pod} all-reduce({pod} %s), channel_id=3, replica_groups={POD}, {reduce}\n"
        )
    else:
        body = f"  %ar = {tile} negate({tile} %c)\n"
    return (
        "HloModule admission_test, entry_computation_layout={(bf16[8,128]{1,0})->bf16[8,128]{1,0}}, "
        f"{header}\n\n"
        f"%add (x: {dtype}[], y: {dtype}[]) -> {dtype}[] {{\n  %x = {dtype}[] parameter(0)\n"
        f"  %y = {dtype}[] parameter(1)\n  ROOT %sum = {dtype}[] add({dtype}[] %x, {dtype}[] %y)\n}}\n\n"
        "ENTRY %main (p0: bf16[8,128]) -> bf16[8,128] {\n  %p0 = bf16[8,128]{1,0} parameter(0)\n"
        f"  %c = {tile} convert(bf16[8,128]{{1,0}} %p0)\n{body}{extra}"
        f"  ROOT %out = bf16[8,128]{{1,0}} convert({tile} %ar)\n}}\n"
    )


def test_physical_axes_with_a_4_kib_full_pod_payload_are_admitted():
    assert check_hlo_collectives(optimized_hlo()) == dict(
        passed=True,
        profile="research_ws32_axis_payload_v1",
        num_partitions=32,
        instructions=10,
        collectives={"all-reduce": 2, "all-gather": 1},
        maximum_collective_payload_bytes=32768,
        frozen_graph_admission_inherited=False,
    )


def test_bf16_halves_the_payload_of_the_same_shapes():
    report = check_hlo_collectives(optimized_hlo(dtype="bf16", pod_elements=2048))
    assert report["passed"] and report["maximum_collective_payload_bytes"] == 16384


A2A = (
    "  %a2a = f32[8,128]{1,0} all-to-all(f32[8,128]{1,0} %ar), channel_id=4, "
    f"replica_groups={FEATURE}, dimensions={{0}}, use_global_device_ids=true\n"
)
# 33,554,433 f32 values: one element over 128 MiB, the limit for every collective.
OVER_128_MIB = (
    "  %big = f32[33554433]{0} all-reduce(f32[33554433]{0} %c), channel_id=5, "
    f"replica_groups={FEATURE}, use_global_device_ids=true, to_apply=%add\n"
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (optimized_hlo(header="num_partitions=16"), "model graph requires exactly 32 partitions/one replica"),
        (
            optimized_hlo(header="num_partitions=32, replica_count=2"),
            "model graph requires exactly 32 partitions/one replica",
        ),
        (optimized_hlo(extra=A2A), "unreviewed collective kind in model graph: all-to-all"),
        (optimized_hlo(feature_groups=PAIRS), "collective does not follow physical expert8/feature4 axes"),
        (optimized_hlo(pod_elements=1025), "oversized or unparsed collective in model graph"),
        (optimized_hlo(extra=OVER_128_MIB), "oversized or unparsed collective in model graph"),
        (optimized_hlo(dtype="c64"), "unknown collective dtype"),
        (optimized_hlo(collectives=False), "model graph has no parsed collectives"),
    ],
    ids=[
        "partitions",
        "replicas",
        "all_to_all",
        "non_physical_groups",
        "pod_4100_bytes",
        "over_128_mib",
        "dtype",
        "none",
    ],
)
def test_refusals_and_their_messages(text, message):
    with pytest.raises(ValueError) as refused:
        check_hlo_collectives(text)
    assert str(refused.value) == message
