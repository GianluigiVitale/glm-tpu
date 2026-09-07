"""CPU-only evidence/protocol tests; no TPU initialization."""

import copy
from hashlib import sha256
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import probe_ws32_prefill_moe as probe
from scripts.greenfield import ws32_prefill_moe_campaign as campaign


def test_cases_preserve_captured_row_and_concentrated_owner():
    hidden = np.ones((1, 6144), dtype=ml_dtypes.bfloat16)
    routes = np.arange(128, 136, dtype=np.int32)[None, :]
    weights = np.arange(1, 9, dtype=np.float32)[None, :] / 36
    for case in probe.CASES:
        h, r, w = probe.case_rows(hidden, routes, weights, case)
        np.testing.assert_array_equal(h[:1], hidden)
        np.testing.assert_array_equal(r[:1], routes)
        np.testing.assert_array_equal(w[:1], weights)
        assert np.all(h[4] == 0) and np.any(h[1] != hidden[0])
        assert h.shape == (17, 6144) and h.dtype == hidden.dtype
        assert all(len(np.unique(row)) == 8 for row in r)
        if case == "concentrated":
            assert np.all(r // 32 == 4)
            for i in range(17):
                assert dict(zip(r[i], w[i])) == dict(zip(routes[0], weights[0]))
        else:
            assert len(np.unique(r // 32)) == 8
    with pytest.raises(ValueError):
        probe.case_rows(
            hidden, routes - np.arange(8)[None, :] * 32, weights, "concentrated"
        )


def valid_hlo():
    feature = (
        "{{"
        + "},{".join(",".join(str(e * 4 + f) for f in range(4)) for e in range(8))
        + "}}"
    )
    expert = (
        "{{"
        + "},{".join(",".join(str(e * 4 + f) for e in range(8)) for f in range(4))
        + "}}"
    )
    lines = ["HloModule admission", "ENTRY %main {"]
    for n, dims in enumerate(("32,2048,1536", "32,2048,1536", "32,1536,2048")):
        lines.append(
            f'%g{n} = f32[136,1536] custom-call(u8[{dims}] %w{n}), custom_call_target="tpu_custom_call", metadata={{op_name="greenfield_prefill_grouped_raw_fp8"}}'
        )
    for n in range(3):
        lines.append(
            f'%s{n} = bf16[17,1536] custom-call(%x), custom_call_target="tpu_custom_call", metadata={{op_name="greenfield_fp8_block_matmul"}}'
        )
    lines.extend(
        [
            f"%f = bf16[17,1536] all-reduce(%x), replica_groups={feature}",
            f"%e = bf16[17,1536] all-reduce(%f), replica_groups={expert}",
            "}",
        ]
    )
    return "\n".join(lines)


def test_hlo_refuses_full_pod_hidden_or_weight_expansion():
    hlo = valid_hlo()
    assert probe.check_hlo(hlo)["passed"]
    for bad in (
        hlo.replace("all-reduce", "all-gather"),
        hlo.replace("u8[32,2048,1536]", "bf16[32,2048,1536]"),
        hlo + "\n%full = f32[100663296] copy(%x)",
        hlo.replace("greenfield_prefill_grouped_raw_fp8", "unknown"),
    ):
        assert not probe.check_hlo(bad)["passed"]


def valid_workers():
    records = []
    digest = "a" * 64
    for rank in range(8):
        slots = list(range(rank * 4, rank * 4 + 4))
        records.append(
            dict(
                status="SUCCESS",
                protocol=probe.PROTOCOL,
                code_hash="b" * 40,
                admission_only=True,
                performance_claim=False,
                iterations=0,
                latency=None,
                rows=17,
                launch_rank=rank,
                jax_process_index=rank,
                hostname=f"host-w-{rank}",
                pid=123,
                start_ticks=567,
                boot_id="boot",
                packed_manifest_sha256=probe.PACK_SHA,
                oracle_manifest_sha256=probe.ORACLE_SHA,
                mesh_sha256=probe.MESH_SHA,
                topology_sha256=probe.TOPOLOGY_SHA,
                topology_fleet_sha256=probe.FLEET_SHA,
                physical_device_ids=np.arange(32).reshape(8, 4).tolist(),
                hlo={"sha256": digest, "contract": {"passed": True}},
                compiled_memory_estimate={
                    "argument_size_in_bytes": 1000,
                    "output_size_in_bytes": 1000,
                    "temp_size_in_bytes": 1000,
                },
                local_device_slots=[
                    dict(device_slot=s, device_id=s, file_sha256=digest) for s in slots
                ],
                device_memory_stats_including_reference=[
                    dict(
                        device_id=s,
                        stats={"peak_bytes_in_use": 4000, "bytes_limit": 33014413312},
                    )
                    for s in slots
                ],
                cases={
                    case: dict(
                        passed=True,
                        fleet_passed=True,
                        reference_vs_legacy={"passed": True},
                        input_sha256=[digest] * 3,
                        shards=[
                            dict(
                                device_slot=s,
                                device_id=s,
                                passed=True,
                                bit_mismatches=0,
                                output_sha256=digest,
                                reference_sha256=digest,
                            )
                            for s in slots
                        ],
                    )
                    for case in probe.CASES
                },
            )
        )
    return records


def test_fleet_refuses_incoherent_inputs_missing_owners_memory_and_timing():
    records = valid_workers()
    campaign.validate_workers(records, "b" * 40)
    for mutate in (
        lambda r: r[0].update(performance_claim=True),
        lambda r: r[0].update(iterations=1),
        lambda r: r[0].update(launch_rank=1),
        lambda r: r[0]["cases"]["normal"].update(input_sha256=["c" * 64] * 3),
        lambda r: r[0]["local_device_slots"][0].update(device_id=3),
        lambda r: r[0]["device_memory_stats_including_reference"][0]["stats"].update(
            peak_bytes_in_use=33014413312
        ),
        lambda r: r[0]["compiled_memory_estimate"].update(temp_size_in_bytes=1024**3),
        lambda r: r[0]["cases"]["normal"]["shards"][0].update(bit_mismatches=1),
    ):
        changed = copy.deepcopy(records)
        mutate(changed)
        with pytest.raises(ValueError):
            campaign.validate_workers(changed, "b" * 40)


def test_original_output_bytes_and_reference_hlo_are_checked(tmp_path):
    hlo = valid_hlo()
    (tmp_path / "candidate.optimized_hlo.txt").write_text(hlo)
    (tmp_path / "reference.optimized_hlo.txt").write_text("reference")
    values = np.full((17, 1536), 0x3F80, np.uint16)
    digest = sha256(values.tobytes()).hexdigest()
    record = valid_workers()[0]
    record["hlo"]["sha256"] = sha256(hlo.encode()).hexdigest()
    record["reference_hlo_sha256"] = sha256(b"reference").hexdigest()
    for case in probe.CASES:
        arrays = {}
        for s in record["cases"][case]["shards"]:
            s.update(output_sha256=digest, reference_sha256=digest)
            arrays[f"actual_{s['device_id']}"] = values
            arrays[f"reference_{s['device_id']}"] = values
        np.savez_compressed(tmp_path / f"{case}.npz", **arrays)
    campaign.validate_files(tmp_path, record)
    (tmp_path / "reference.optimized_hlo.txt").write_text("wrong")
    with pytest.raises(ValueError, match="reference HLO"):
        campaign.validate_files(tmp_path, record)


def test_wrapper_dispatches_fleet_before_single_process_bounds():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/run_fp8_matmul_microbench.sh"
    ).read_text()
    assert source.index("ws32_prefill_moe_campaign campaign") < source.index(
        "TPU_PROCESS_BOUNDS=1,1,1"
    )
    assert "validate_record(runner, pin, boundary=boundary)" in source
    assert "real_layer3_b17_arithmetic_v1_normal_concentrated" in source
    with pytest.raises(ValueError):
        campaign.run_root("../../other")
