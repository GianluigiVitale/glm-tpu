from scripts.collective_chain_microbench import (
    _chain_body,
    _hlo_collective_counts,
    _percentile,
)


def test_hlo_collective_counts_handles_sync_and_async_pairs():
    hlo = """
  %ar0 = bf16[2,6144] all-reduce(%x), replica_groups={{0,1}}
  %ar1 = bf16[2,6144] all-reduce-start(%ar0), replica_groups={{0,1}}
  %ar2 = bf16[2,6144] all-reduce-done(%ar1)
  %ag0 = bf16[4,6144] all-gather(%x), dimensions={0}
  %rs0 = bf16[1,6144] reduce-scatter-start(%x), dimensions={0}
  %rs1 = bf16[1,6144] reduce-scatter-done(%rs0)
  %barrier = bf16[2,6144] optimization-barrier(%ar2)
"""

    counts = _hlo_collective_counts(hlo)

    assert counts["all_reduce_sync"] == 1
    assert counts["all_reduce_start"] == 1
    assert counts["all_reduce_done"] == 1
    assert counts["logical_all_reduce"] == 2
    assert counts["all_gather_sync"] == 1
    assert counts["reduce_scatter_start"] == 1
    assert counts["reduce_scatter_done"] == 1
    assert counts["optimization_barrier"] == 1


def test_percentile_interpolates_without_external_dependencies():
    values = [4.0, 1.0, 3.0, 2.0]

    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 0.5) == 2.5
    assert _percentile(values, 1.0) == 4.0


def test_unknown_collective_chain_kind_is_rejected():
    try:
        _chain_body("not-a-layout")
    except ValueError as error:
        assert str(error) == "not-a-layout"
    else:
        raise AssertionError("unknown collective-chain layout was accepted")
