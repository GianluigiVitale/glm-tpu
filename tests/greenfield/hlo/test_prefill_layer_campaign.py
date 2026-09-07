"""Fail-closed layer HLO/fleet admission, with no cloud or TPU access."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.greenfield.prefill_layer_hlo import EXPERT, FEATURE, check_layer_hlo
from scripts.greenfield import ws32_prefill_layer_campaign as campaign


def valid_layer0_hlo():
    def groups(values):
        return "{{" + "},{".join(",".join(map(str, g)) for g in values) + "}}"

    lines = ["HloModule full_layer", "ENTRY %main {", "%w = u8[128,128] parameter(0)"]
    number = 0
    reductions = [
        (4, "f32", (17,)),
        (4, "f32", (17,)),
        (4, "f32", (17, 2048)),
        (4, "f32", (17, 576)),
        (4, "f32", (17, 4)),
        (4, "f32", (17, 128)),
        (4, "f32", (2, 17, 1536)),
        (8, "bf16", (17, 2048, 640)),
        (8, "f32", (17, 1536)),
        (8, "f32", (17, 1536)),
    ]
    gathers = [
        (8, "f32", (8, 17, 516)),
        (8, "s32", (8, 17, 2048)),
        (8, "f32", (8, 17, 2048)),
        (4, "bf16", (17, 6144)),
    ]
    for opcode, items in (("all-reduce", reductions), ("all-gather", gathers)):
        for size, dtype, shape in items:
            dims = ",".join(map(str, shape))
            lines.append(f"%input{number} = {dtype}[{dims}] parameter({number+1})")
            lines.append(
                f"%c{number} = {dtype}[{dims}] {opcode}(%input{number}), replica_groups={groups(FEATURE if size==4 else EXPERT)}"
            )
            number += 1
    for number, name in enumerate(
        ["greenfield_fp8_block_matmul"] * 9
        + [
            "greenfield_fp8_structured_kv_b_q_absorb",
            "greenfield_fp8_structured_kv_b_value",
            "greenfield_pregathered_sparse_mla_prefill_m17",
        ]
    ):
        lines.append(
            f'%call{number} = bf16[17,1536] custom-call(%w), custom_call_target="tpu_custom_call", metadata={{op_name="{name}"}}'
        )
    return "\n".join(lines + ["}"])


def test_full_layer_hlo_inventory_is_distinct_from_moe():
    proof = check_layer_hlo(valid_layer0_hlo(), layer=0)
    assert proof["passed"], proof["checks"]
    assert len(proof["collectives"]) == 14
    assert len(proof["custom_calls"]) == 12


@pytest.mark.parametrize(
    "old,new",
    [
        ("{0,1,2,3}", "{0,1,2,7}"),
        ("all-reduce", "all-to-all"),
        ("f32[17,2048]", "f32[17,1024]"),
        ("prefill_m17", "decode_m1"),
        ("tpu_custom_call", "host_callback"),
        ("u8[128,128]", "bf16[256,2048,6144]"),
    ],
)
def test_layer_hlo_refuses_wrong_groups_payload_deadrow_and_expansion(old, new):
    assert not check_layer_hlo(valid_layer0_hlo().replace(old, new), layer=0)["passed"]


def test_reduction_input_dtype_is_bound_despite_fused_output_cast():
    hlo = valid_layer0_hlo().replace("%c8 = f32[17,1536]", "%c8 = bf16[17,1536]")
    assert check_layer_hlo(hlo, layer=0)["passed"]
    hlo = hlo.replace("%input8 = f32[17,1536]", "%input8 = bf16[17,1536]")
    assert not check_layer_hlo(hlo, layer=0)["passed"]


def valid_workers(layer=0):
    pin = "a" * 40
    pins = {"fixture": "metadata only"}
    ledger = {
        s: {"full_sha256": "b" * 64, "selected": {"test.weight_bits": "c" * 64}}
        for s in range(32)
    }
    records = []
    for rank in range(8):
        local = list(range(rank * 4, rank * 4 + 4))
        programs = {
            name: {
                "stablehlo_sha256": "d" * 64,
                "optimized_hlo_sha256": "e" * 64,
                "compiled_memory": dict(
                    argument_size_in_bytes=1,
                    output_size_in_bytes=1,
                    temp_size_in_bytes=1,
                ),
            }
            for name in campaign.program_names(layer)
        }
        records.append(
            dict(
                status="SUCCESS",
                protocol=campaign.PROTOCOL,
                code_hash=pin,
                layer=layer,
                launch_rank=rank,
                hostname=f"worker{rank}",
                jax_process_index=7 - rank,
                physical_device_ids=list(reversed(range(32))),
                selected_layer_ids=[layer],
                admission_only=True,
                performance_claim=False,
                iterations=0,
                latency=None,
                rows=17,
                reference_scope="RAW_SCALAR_NOT_PROMOTED_DECODER_OR_LEGACY",
                state_scope="REAL_WEIGHTS_SYNTHETIC_PREFIX_AND_ACTIVATIONS",
                integrity_scope="selected_layer_tensors_only_not_complete_checkpoint",
                checkpoint_pins=pins,
                payload_bytes_per_chip=campaign.PAYLOAD_BYTES[layer],
                mesh_sha256=campaign.MESH_SHA,
                topology_sha256=campaign.TOPOLOGY_SHA,
                topology_fleet_sha256=campaign.FLEET_SHA,
                pid=100 + rank,
                start_ticks=200,
                boot_id="boot",
                programs=programs,
                hlo={"contract": {"passed": True}},
                local_device_slots=[
                    dict(
                        device_slot=s,
                        device_id=31 - s,
                        observed_selected_tensor_sha256=ledger[s]["selected"],
                        expected_full_file_sha256_not_verified=ledger[s]["full_sha256"],
                        selected_payload_bytes=campaign.PAYLOAD_BYTES[layer],
                    )
                    for s in local
                ],
                device_memory_stats_including_reference=[
                    dict(
                        device_id=31 - s,
                        stats=dict(peak_bytes_in_use=100, bytes_limit=1000),
                    )
                    for s in local
                ],
                cases={
                    case: dict(
                        passed=True,
                        input_sha256={"test": "x"},
                        replay=dict(
                            passed=True, owners={str(31 - s): {} for s in local}
                        ),
                    )
                    for case in campaign.CASES
                },
            )
        )
    return records, pin, pins, ledger


@pytest.mark.parametrize("layer", [0, 3])
def test_fleet_admission_binds_all_owners_not_host_rank(layer):
    records, pin, pins, ledger = valid_workers(layer)
    campaign.validate_workers(records, pin, layer=layer, pins=pins, ledger=ledger)


@pytest.mark.parametrize(
    "mutation",
    [
        "rank",
        "process",
        "host",
        "code",
        "scope",
        "layer",
        "pin",
        "tensor",
        "slot",
        "bytes",
        "memory",
        "headroom",
        "case",
        "program",
        "graph",
        "input",
        "order",
    ],
)
def test_fleet_admission_refuses_drift(mutation):
    records, pin, pins, ledger = deepcopy(valid_workers())
    r = records[0]
    if mutation == "rank":
        r["launch_rank"] = 1
    elif mutation == "process":
        r["jax_process_index"] = 0
    elif mutation == "host":
        r["hostname"] = "worker1"
    elif mutation == "code":
        r["code_hash"] = "f" * 40
    elif mutation == "scope":
        r["integrity_scope"] = "complete_checkpoint"
    elif mutation == "layer":
        r["selected_layer_ids"] = [0, 3]
    elif mutation == "pin":
        r["checkpoint_pins"] = {}
    elif mutation == "tensor":
        r["local_device_slots"][0]["observed_selected_tensor_sha256"] = {}
    elif mutation == "slot":
        r["local_device_slots"][0]["device_slot"] = 4
    elif mutation == "bytes":
        r["local_device_slots"][0]["selected_payload_bytes"] += 1
    elif mutation == "memory":
        r["programs"]["candidate"]["compiled_memory"]["temp_size_in_bytes"] = (
            3 * 1024**3
        )
    elif mutation == "headroom":
        r["device_memory_stats_including_reference"][0]["stats"][
            "peak_bytes_in_use"
        ] = 1000
    elif mutation == "case":
        del r["cases"]["tail"]
    elif mutation == "program":
        r["programs"]["surprise"] = {}
    elif mutation == "graph":
        r["programs"]["repair"]["stablehlo_sha256"] = "f" * 64
    elif mutation == "input":
        r["cases"]["empty"]["input_sha256"] = {}
    elif mutation == "order":
        r["physical_device_ids"] = list(range(32))
    with pytest.raises((ValueError, KeyError)):
        campaign.validate_workers(records, pin, layer=0, pins=pins, ledger=ledger)


def test_bounded_wrapper_dispatches_layer_before_generic_admission():
    source = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    dispatch = source.index(
        "-m scripts.greenfield.ws32_prefill_layer_campaign campaign"
    )
    assert (
        source.index("strict_census pre")
        < dispatch
        < source.index("scripts/greenfield/probe_prefill_grouped_fp8.py")
    )
    assert "GLM_GREENFIELD_PREFILL_LAYER" in source
    assert (
        "if fleet_layer:\n    from scripts.greenfield.ws32_prefill_layer_campaign import validate_record"
        in source
    )
    assert "latency_ms=None if untimed" in source
    assert "exec 9>/home/gianl/glm-run/.glm_pod_workload.lock" in source
    assert "exec 8>/home/gianl/.glm-tpu-rsync.lock" in source
