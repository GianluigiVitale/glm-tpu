from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    STRATEGY_ND_ALGORITHM,
    _BALANCED_FOUR_WAY_TREES,
    _candidate_output_bits,
    bfloat16_bits_to_float32,
)
from glm_tpu.greenfield.benchmarking.m2048_association_fingerprint import (
    M2048_ROWS,
    M2048_SEED,
    M2048_TRIALS,
    M2048_WIDTH,
    M2048StrategyNdFingerprintConfig,
    analyze_m2048_row0_association,
    generate_m2048_row0_input_bits,
    validate_m2048_strategy_nd_fingerprint_hlo,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


MEMBERS = tuple(range(32))


def _coordinates() -> dict[int, tuple[int, int, int]]:
    return {
        device_id: (device_id % 2, (device_id // 2) % 4, device_id // 8)
        for device_id in MEMBERS
    }


def _m2048_hlo() -> str:
    group = ",".join(map(str, MEMBERS))
    backend = json.dumps(
        {"collective_algorithm_config": STRATEGY_ND_ALGORITHM},
        separators=(",", ":"),
    )
    return f'''HloModule m2048_fingerprint, num_partitions=32, replica_count=1

add {{
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT sum = bf16[] add(x, y)
}}

ENTRY main {{
  input = u16[1,2048,6144]{{2,1,0:T(8,128)(2,1)}} parameter(0)
  flat = u16[2048,6144]{{1,0:T(8,128)(2,1)}} reshape(input)
  payload = bf16[2048,6144]{{1,0:T(8,128)(2,1)}} bitcast-convert(flat)
  reduced = bf16[2048,6144]{{1,0:T(8,128)(2,1)S(3)}} all-reduce(payload), replica_groups={{{{{group}}}}}, use_global_device_ids=true, to_apply=add, metadata={{op_name="jit(fingerprint)/shard_map/m2048_strategy_nd_association_fingerprint/psum"}}, backend_config={backend}
  full = u16[2048,6144]{{1,0:T(8,128)(2,1)}} bitcast-convert(reduced)
  row = bf16[1,6144]{{1,0:T(8,128)(2,1)}} slice(reduced), slice={{[0:1], [0:6144]}}
  row-flat = bf16[6144]{{0:T(1024)(128)(2,1)}} reshape(row)
  row-bits = u16[6144]{{0:T(1024)(128)(2,1)}} bitcast-convert(row-flat)
  ROOT result = (u16[2048,6144]{{1,0:T(8,128)(2,1)}}, u16[6144]{{0:T(1024)(128)(2,1)}}) tuple(full, row-bits)
}}
'''


def _m2048_fused_row_hlo() -> str:
    hlo = _m2048_hlo()
    fusion = '''
row_fusion {
  p = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} parameter(0)
  s = bf16[1,6144]{1,0:T(8,128)(2,1)} slice(p), slice={[0:1], [0:6144]}
  flat = bf16[6144]{0:T(1024)(128)(2,1)} reshape(s)
  ROOT bits = u16[6144]{0:T(1024)(128)(2,1)} bitcast-convert(flat)
}

'''
    hlo = hlo.replace("ENTRY main {", fusion + "ENTRY main {")
    old = '''  row = bf16[1,6144]{1,0:T(8,128)(2,1)} slice(reduced), slice={[0:1], [0:6144]}
  row-flat = bf16[6144]{0:T(1024)(128)(2,1)} reshape(row)
  row-bits = u16[6144]{0:T(1024)(128)(2,1)} bitcast-convert(row-flat)'''
    return hlo.replace(
        old,
        "  row-bits = u16[6144]{0:T(1024)(128)(2,1)} "
        "fusion(reduced), kind=kLoop, calls=row_fusion",
    )


def test_config_and_input_bank_are_exact_and_deterministic() -> None:
    config = M2048StrategyNdFingerprintConfig()
    assert config.to_dict() == {
        "rows": M2048_ROWS,
        "seed": M2048_SEED,
        "trials": M2048_TRIALS,
        "width": M2048_WIDTH,
    }
    first = generate_m2048_row0_input_bits(config)
    second = generate_m2048_row0_input_bits(config)
    assert first.shape == (32, 32, 6144)
    assert first.dtype == np.uint16
    np.testing.assert_array_equal(first, second)

    with pytest.raises(BenchmarkValidationError, match="protected M2048"):
        generate_m2048_row0_input_bits(
            M2048StrategyNdFingerprintConfig(trials=31)
        )


def test_hlo_contract_accepts_only_the_exact_live_m2048_collective() -> None:
    hlo = _m2048_hlo()
    report, algorithm = validate_m2048_strategy_nd_fingerprint_hlo(hlo, MEMBERS)
    assert report.valid
    assert algorithm == STRATEGY_ND_ALGORITHM

    mutations = (
        (
            hlo.replace("bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} all-reduce", "bf16[2048,6144]{1,0} all-reduce"),
            "TPU result layout",
        ),
        (hlo.replace('"strategy":"StrategyND"', '"strategy":"StrategyRing"'), "pinned StrategyND"),
        (hlo.replace("ROOT sum = bf16[] add(x, y)", "ROOT sum = bf16[] maximum(x, y)"), "exact scalar BF16 add"),
        (hlo.replace("slice={[0:1], [0:6144]}", "slice={[1:2], [0:6144]}"), "row-zero"),
        (hlo.replace("all-reduce(payload)", "all-reduce(flat)"), "operand/result"),
        (hlo.replace("tuple(full, row-bits)", "tuple(full, full)"), "root order|dead or alternate instructions"),
    )
    for drifted, message in mutations:
        with pytest.raises(BenchmarkValidationError, match=message):
            validate_m2048_strategy_nd_fingerprint_hlo(drifted, MEMBERS)

    rogue = hlo.replace(
        "  reduced = bf16[2048,6144]",
        "  rogue = bf16[2048,6144]{1,0:T(8,128)(2,1)} negate(payload)\n"
        "  reduced = bf16[2048,6144]",
    ).replace("all-reduce(payload)", "all-reduce(rogue)")
    with pytest.raises(BenchmarkValidationError, match="source arithmetic"):
        validate_m2048_strategy_nd_fingerprint_hlo(rogue, MEMBERS)

    post_add = hlo.replace(
        "  full = u16[2048,6144]",
        "  contaminated = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} add(reduced, reduced)\n"
        "  full = u16[2048,6144]",
    ).replace("bitcast-convert(reduced)\n  row =", "bitcast-convert(contaminated)\n  row =")
    with pytest.raises(BenchmarkValidationError, match="unary representation"):
        validate_m2048_strategy_nd_fingerprint_hlo(post_add, MEMBERS)

    with pytest.raises(BenchmarkValidationError, match="sorted global"):
        validate_m2048_strategy_nd_fingerprint_hlo(hlo, tuple(reversed(MEMBERS)))


def test_fused_row0_contract_rejects_dead_alternate_and_mismapped_slices() -> None:
    hlo = _m2048_fused_row_hlo()
    report, _ = validate_m2048_strategy_nd_fingerprint_hlo(hlo, MEMBERS)
    assert report.valid

    dead_correct_slice = hlo.replace(
        "  flat = bf16[6144]",
        "  wrong = bf16[1,6144]{1,0:T(8,128)(2,1)} "
        "slice(p), slice={[1:2], [0:6144]}\n"
        "  flat = bf16[6144]",
    ).replace("reshape(s)", "reshape(wrong)")
    with pytest.raises(BenchmarkValidationError, match="dead or alternate"):
        validate_m2048_strategy_nd_fingerprint_hlo(dead_correct_slice, MEMBERS)

    extra_operand = hlo.replace(
        "fusion(reduced), kind=kLoop",
        "fusion(reduced, reduced), kind=kLoop",
    )
    with pytest.raises(BenchmarkValidationError, match="alternate operand"):
        validate_m2048_strategy_nd_fingerprint_hlo(extra_operand, MEMBERS)

    wrong_mapping = hlo.replace(
        "parameter(0)\n  s = bf16[1,6144]",
        "parameter(1)\n  s = bf16[1,6144]",
    )
    with pytest.raises(BenchmarkValidationError, match="structure drifted"):
        validate_m2048_strategy_nd_fingerprint_hlo(wrong_mapping, MEMBERS)

    alternate_root = hlo.replace(
        "  flat = bf16[6144]",
        "  wrong = bf16[1,6144]{1,0:T(8,128)(2,1)} "
        "slice(p), slice={[1:2], [0:6144]}\n"
        "  wrong-flat = bf16[6144]{0:T(1024)(128)(2,1)} reshape(wrong)\n"
        "  wrong-bits = u16[6144]{0:T(1024)(128)(2,1)} "
        "bitcast-convert(wrong-flat)\n"
        "  flat = bf16[6144]",
    ).replace("ROOT bits =", "bits =").replace(
        "\n}\n\nENTRY main", "\n  ROOT alternate = u16[6144]{0:T(1024)(128)(2,1)} copy(wrong-bits)\n}\n\nENTRY main"
    )
    with pytest.raises(BenchmarkValidationError, match="dead or alternate"):
        validate_m2048_strategy_nd_fingerprint_hlo(alternate_root, MEMBERS)


def test_exact_m2048_bank_recovers_one_declared_candidate_per_lane() -> None:
    inputs = generate_m2048_row0_input_bits(M2048StrategyNdFingerprintConfig())
    decoded = bfloat16_bits_to_float32(inputs)
    coordinates = _coordinates()
    topology = np.empty((32, 4, 2, 4, 6144), dtype=np.float32)
    for member_index, device_id in enumerate(MEMBERS):
        x, y, z = coordinates[device_id]
        topology[:, y, x, z, :] = decoded[:, member_index, :]
    outputs = _candidate_output_bits(
        topology,
        (0, 1, 2),
        {0: _BALANCED_FOUR_WAY_TREES[0], 2: _BALANCED_FOUR_WAY_TREES[0]},
    )
    analysis = analyze_m2048_row0_association(
        inputs, outputs, MEMBERS, coordinates
    )
    assert analysis["every_lane_has_exactly_one_candidate"] is True
    assert analysis["column_candidate_count_histogram"] == {"1": 6144}
    assert analysis["declared_candidate_family_count"] == 54

    drifted = outputs.copy()
    drifted[:, 0] ^= np.uint16(1)
    with pytest.raises(BenchmarkValidationError, match="not unique"):
        analyze_m2048_row0_association(inputs, drifted, MEMBERS, coordinates)


PROTECTED_V10_HLO = (
    Path(__file__).parent / "fixtures" / "m2048_v10_protected_optimized_hlo.txt"
)
PROTECTED_V10_HLO_SHA256 = (
    "a96ff87a934afeac31df9218bb4436230924584c4775d8b4f943d68a20c39ab2"
)


def _protected_v10_hlo() -> str:
    raw = PROTECTED_V10_HLO.read_bytes()
    assert sha256(raw).hexdigest() == PROTECTED_V10_HLO_SHA256
    assert len(raw) == 8311
    return raw.decode("ascii")


def test_protected_v10_tpu_hlo_unit_extent_u16_reduce_is_accepted() -> None:
    """The exact bytes that V10 false-rejected before numerics must now pass."""

    hlo = _protected_v10_hlo()
    assert "reduce(%bitcast_convert_type.23, %constant.3), dimensions={0}" in hlo
    report, algorithm = validate_m2048_strategy_nd_fingerprint_hlo(hlo, MEMBERS)
    assert report.valid
    assert algorithm == STRATEGY_ND_ALGORITHM
    assert algorithm["emitter"] == "RotatedPincerEmitter"


def _v10_mutation(hlo: str, old: str, new: str) -> str:
    assert hlo.count(old) == 1, old
    return hlo.replace(old, new)


REDUCE_LINE = (
    "ROOT %reduce.1 = u16[6144]{0:T(1024)(128)(2,1)} "
    "reduce(%bitcast_convert_type.23, %constant.3), dimensions={0}, "
    "to_apply=%bitcast_convert_type.18.reduce_sub_computation"
)
SLICE_LINE = (
    "%slice.7 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice(%param_0.7), "
    "slice={[0:1], [0:6144]}"
)
BITCAST_LINE = (
    "%bitcast_convert_type.23 = u16[1,6144]{1,0:T(2,128)(2,1)} "
    "bitcast-convert(%slice.7)"
)
CONSTANT_LINE = "%constant.3 = u16[]{:T(256)} constant(0)"
REDUCER_ROOT = "ROOT %add.2 = u16[] add(%lhs, %rhs)"


@pytest.mark.parametrize(
    ("label", "old", "new", "message"),
    (
        (
            "nonzero initializer",
            CONSTANT_LINE,
            "%constant.3 = u16[]{:T(256)} constant(1)",
            "exact u16 zero",
        ),
        (
            "initializer is a parameter-shaped non-constant",
            CONSTANT_LINE,
            "%constant.3 = u16[]{:T(256)} copy(%param_0.7)",
            "exact u16 zero",
        ),
        (
            "wrong reduced dimension",
            REDUCE_LINE,
            REDUCE_LINE.replace("dimensions={0}", "dimensions={1}"),
            "shape/dimension drifted",
        ),
        (
            "missing dimensions attribute",
            REDUCE_LINE,
            REDUCE_LINE.replace(", dimensions={0}", ""),
            "dimensions attribute is not unique|shape/dimension drifted",
        ),
        (
            "swapped reduce operands",
            REDUCE_LINE,
            REDUCE_LINE.replace(
                "reduce(%bitcast_convert_type.23, %constant.3)",
                "reduce(%constant.3, %bitcast_convert_type.23)",
            ),
            "shape/dimension drifted",
        ),
        (
            "variadic reduce",
            REDUCE_LINE,
            REDUCE_LINE.replace(
                "reduce(%bitcast_convert_type.23, %constant.3)",
                "reduce(%bitcast_convert_type.23, %bitcast_convert_type.23, "
                "%constant.3, %constant.3)",
            ),
            "one value and one initializer",
        ),
        (
            "reduce result keeps a dimension",
            REDUCE_LINE,
            REDUCE_LINE.replace(
                "ROOT %reduce.1 = u16[6144]{0:T(1024)(128)(2,1)}",
                "ROOT %reduce.1 = u16[1,6144]{1,0:T(2,128)(2,1)}",
            ),
            "shape/dimension drifted",
        ),
        (
            "reducer maximum",
            REDUCER_ROOT,
            "ROOT %add.2 = u16[] maximum(%lhs, %rhs)",
            "exact scalar u16 add",
        ),
        (
            "reducer multiply",
            REDUCER_ROOT,
            "ROOT %add.2 = u16[] multiply(%lhs, %rhs)",
            "exact scalar u16 add",
        ),
        (
            "reducer ignores one parameter",
            REDUCER_ROOT,
            "ROOT %add.2 = u16[] add(%lhs, %lhs)",
            "exact scalar u16 add",
        ),
        (
            "reducer is the BF16 collective reducer",
            REDUCE_LINE,
            REDUCE_LINE.replace(
                "to_apply=%bitcast_convert_type.18.reduce_sub_computation",
                "to_apply=%region_0.0",
            ),
            "exact scalar u16 add",
        ),
        (
            "row one instead of row zero",
            SLICE_LINE,
            SLICE_LINE.replace("slice={[0:1], [0:6144]}", "slice={[1:2], [0:6144]}"),
            "row-zero slice drifted",
        ),
        (
            "reduce bypasses the slice",
            BITCAST_LINE,
            "%bitcast_convert_type.23 = u16[1,6144]{1,0:T(2,128)(2,1)} "
            "bitcast-convert(%param_0.7)",
            "shape/dimension drifted|dead or alternate",
        ),
        (
            "two caller operands",
            "fusion(%psum.7), kind=kLoop, calls=%fused_computation.2",
            "fusion(%psum.7, %psum.7), kind=kLoop, calls=%fused_computation.2",
            "alternate operand|exactly one caller operand",
        ),
        (
            "dead extra slice inside the fusion",
            CONSTANT_LINE,
            "%dead.9 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice(%param_0.7), "
            "slice={[1:2], [0:6144]}\n  " + CONSTANT_LINE,
            "dead or alternate",
        ),
        (
            "duplicate dimensions attribute",
            REDUCE_LINE,
            REDUCE_LINE.replace("dimensions={0}", "dimensions={1}, dimensions={0}"),
            "dimensions attribute is not unique",
        ),
        (
            "literal third reduce operand hidden from the parser",
            REDUCE_LINE,
            REDUCE_LINE.replace(
                "reduce(%bitcast_convert_type.23, %constant.3)",
                "reduce(%bitcast_convert_type.23, %constant.3, {1})",
            ),
            "one value and one initializer",
        ),
        (
            "reduce is not the fusion root",
            REDUCE_LINE,
            REDUCE_LINE.replace("ROOT %reduce.1", "%reduce.1")
            + "\n  ROOT %copy.9 = u16[6144]{0:T(1024)(128)(2,1)} copy(%reduce.1)",
            "not the fusion root",
        ),
        (
            "dead ENTRY instruction",
            "  ROOT %tuple.5 =",
            "  %dead.entry = u16[2048,6144]{1,0:T(8,128)(2,1)} "
            "bitcast-convert(%psum.7)\n  ROOT %tuple.5 =",
            "dead or alternate instructions",
        ),
        (
            "constant inside the pre-collective fusion",
            "ROOT %bitcast.1 = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} "
            "bitcast(%bitcast_convert_type.20)",
            "%stray = u16[]{:T(256)} constant(0)\n  "
            "ROOT %bitcast.1 = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} "
            "bitcast(%bitcast_convert_type.20)",
            "structure drifted",
        ),
    ),
)
def test_protected_v10_hlo_hostile_unit_extent_mutations_reject(
    label: str, old: str, new: str, message: str
) -> None:
    hostile = _v10_mutation(_protected_v10_hlo(), old, new)
    with pytest.raises(BenchmarkValidationError, match=message):
        validate_m2048_strategy_nd_fingerprint_hlo(hostile, MEMBERS)


def test_protected_v10_hlo_rejects_non_unit_extent_payload_association() -> None:
    """A two-row reduce would associate payload rows; it must never be accepted."""

    hlo = _protected_v10_hlo()
    hostile = _v10_mutation(
        hlo,
        SLICE_LINE,
        "%slice.7 = bf16[2,6144]{1,0:T(2,128)(2,1)} slice(%param_0.7), "
        "slice={[0:2], [0:6144]}",
    )
    hostile = _v10_mutation(
        hostile,
        BITCAST_LINE,
        "%bitcast_convert_type.23 = u16[2,6144]{1,0:T(2,128)(2,1)} "
        "bitcast-convert(%slice.7)",
    )
    with pytest.raises(BenchmarkValidationError, match="shape/dimension drifted"):
        validate_m2048_strategy_nd_fingerprint_hlo(hostile, MEMBERS)

    signed = _v10_mutation(hlo, BITCAST_LINE, BITCAST_LINE.replace("u16[1,6144]", "s16[1,6144]"))
    with pytest.raises(BenchmarkValidationError, match="shape/dimension drifted"):
        validate_m2048_strategy_nd_fingerprint_hlo(signed, MEMBERS)

    full_rows = _v10_mutation(
        hlo,
        BITCAST_LINE,
        "%bitcast_convert_type.23 = u16[2048,6144]{1,0:T(8,128)(2,1)} "
        "bitcast-convert(%param_0.7)",
    )
    with pytest.raises(BenchmarkValidationError):
        validate_m2048_strategy_nd_fingerprint_hlo(full_rows, MEMBERS)


def test_synthetic_fused_row0_reduce_form_is_accepted_and_alternate_root_rejected() -> None:
    hlo = _m2048_fused_row_hlo()
    reduce_form = hlo.replace(
        '''row_fusion {
  p = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} parameter(0)
  s = bf16[1,6144]{1,0:T(8,128)(2,1)} slice(p), slice={[0:1], [0:6144]}
  flat = bf16[6144]{0:T(1024)(128)(2,1)} reshape(s)
  ROOT bits = u16[6144]{0:T(1024)(128)(2,1)} bitcast-convert(flat)
}''',
        '''u16_add {
  a = u16[] parameter(0)
  b = u16[] parameter(1)
  ROOT s = u16[] add(a, b)
}

row_fusion {
  p = bf16[2048,6144]{1,0:T(8,128)(2,1)S(3)} parameter(0)
  s = bf16[1,6144]{1,0:T(8,128)(2,1)} slice(p), slice={[0:1], [0:6144]}
  bits2 = u16[1,6144]{1,0:T(8,128)(2,1)} bitcast-convert(s)
  zero = u16[] constant(0)
  ROOT bits = u16[6144]{0:T(1024)(128)(2,1)} reduce(bits2, zero), dimensions={0}, to_apply=u16_add
}''',
    )
    assert reduce_form != hlo
    report, _ = validate_m2048_strategy_nd_fingerprint_hlo(reduce_form, MEMBERS)
    assert report.valid

    alternate_root = reduce_form.replace(
        "  ROOT bits = u16[6144]{0:T(1024)(128)(2,1)} reduce(bits2, zero), dimensions={0}, to_apply=u16_add\n}",
        "  bits = u16[6144]{0:T(1024)(128)(2,1)} reduce(bits2, zero), dimensions={0}, to_apply=u16_add\n"
        "  wrong = bf16[1,6144]{1,0:T(8,128)(2,1)} slice(p), slice={[1:2], [0:6144]}\n"
        "  wrong-flat = bf16[6144]{0:T(1024)(128)(2,1)} reshape(wrong)\n"
        "  ROOT alternate = u16[6144]{0:T(1024)(128)(2,1)} bitcast-convert(wrong-flat)\n}",
    )
    assert alternate_root != reduce_form
    with pytest.raises(BenchmarkValidationError, match="dead or alternate|row-zero slice drifted"):
        validate_m2048_strategy_nd_fingerprint_hlo(alternate_root, MEMBERS)

    chained = reduce_form.replace(
        "  zero = u16[] constant(0)\n",
        "  zero = u16[] constant(0)\n"
        "  bits3 = u16[1,1,6144]{2,1,0:T(8,128)(2,1)} reshape(bits2)\n"
        "  pre = u16[1,6144]{1,0:T(8,128)(2,1)} reduce(bits3, zero), dimensions={0}, to_apply=u16_add\n",
    ).replace("reduce(bits2, zero), dimensions={0}, to_apply=u16_add\n}", "reduce(pre, zero), dimensions={0}, to_apply=u16_add\n}")
    assert chained != reduce_form
    with pytest.raises(BenchmarkValidationError, match="sole row-zero unit-extent removal"):
        validate_m2048_strategy_nd_fingerprint_hlo(chained, MEMBERS)
