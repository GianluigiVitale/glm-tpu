"""Distinct v2 reference evidence contracts, CPU-only."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from scripts.greenfield import prefill_materialized_reference as ref
from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.prefill_layer_evidence import BF16
from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT
from tests.greenfield.hlo.test_prefill_layer_campaign import valid_workers


def test_actual_compile_from_abstract_prefix_then_completed_execution(tmp_path):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("feature",))
    prefix = jax.jit(
        jax.shard_map(
            lambda x: (x.astype(jnp.bfloat16), x.astype(jnp.bfloat16)),
            mesh=mesh,
            in_specs=(P(None, "feature"),),
            out_specs=(P(None, "feature"), P(None, "feature")),
        )
    )
    suffix = jax.jit(
        jax.shard_map(
            lambda x, r, weight: x.astype(jnp.float32) @ weight[0]
            + r.astype(jnp.float32),
            mesh=mesh,
            in_specs=(P(None, "feature"), P(None, "feature"), (P(),)),
            out_specs=P(None, "feature"),
        )
    )
    inputs = (jnp.full((1, 8), 1.001, jnp.float32),)
    weight = (jnp.eye(8, dtype=jnp.float32),)
    record = {}
    pre = compile_program(prefix, inputs, "reference_prefix", tmp_path, record)
    shapes = jax.eval_shape(prefix, *inputs)
    assert shapes[0].dtype == jnp.bfloat16
    mlp = compile_program(
        suffix, (shapes[0], shapes[1], weight), "reference", tmp_path, record
    )
    completed = pre(*inputs)
    jax.block_until_ready(completed)
    actual = mlp(*completed, weight)
    np.testing.assert_array_equal(actual, np.full((1, 8), 2, np.float32))
    assert set(record["programs"]) == {"reference_prefix", "reference"}


def materialized_workers():
    records, pin, pins, ledger = valid_workers(3)
    for r in records:
        r.update(protocol=ref.PROTOCOL, reference_scope=ref.REFERENCE_SCOPE)
        example = next(iter(r["programs"].values()))
        r["programs"] = {n: deepcopy(example) for n in ref.PROGRAMS}
        for name in ("reference_prefix", "reference"):
            r["programs"][name]["hlo_contract"] = {"passed": True}
        r["reference_input_sha256"] = {c: "f" * 64 for c in campaign.CASES}
    return records, pin, pins, ledger


def test_v2_is_separate_from_original_and_diagnostic():
    tag = f"greenfield_fp8_{ref.KERNEL}_l3_test"
    assert ref.is_materialized_tag(tag)
    assert not ref.is_materialized_tag(tag.replace("l3", "l0"))
    assert campaign.layer_from_tag(tag) == 3
    files = campaign.evidence_files(3, materialized=True)
    assert "reference_prefix.optimized_hlo.txt" in files
    assert {f"{case}.reference_input.npz" for case in campaign.CASES} <= set(files)
    with pytest.raises(ValueError):
        campaign.evidence_files(0, materialized=True)
    with pytest.raises(ValueError):
        campaign.evidence_files(3, materialized=True, diagnostic=True)
    records, pin, pins, ledger = materialized_workers()
    campaign.validate_workers(
        records, pin, layer=3, pins=pins, ledger=ledger, materialized=True
    )
    with pytest.raises(ValueError):
        campaign.validate_workers(records, pin, layer=3, pins=pins, ledger=ledger)


@pytest.mark.parametrize(
    "mutation", ("scope", "protocol", "capture", "prefix", "hlo", "memory", "case")
)
def test_materialized_worker_refuses_incomplete_or_relabelled_evidence(mutation):
    records, pin, pins, ledger = materialized_workers()
    r = records[0]
    if mutation == "scope":
        r["reference_scope"] = "RAW_SCALAR_NOT_PROMOTED_DECODER_OR_LEGACY"
    elif mutation == "protocol":
        r["protocol"] = campaign.PROTOCOL
    elif mutation == "capture":
        del r["reference_input_sha256"]["tail"]
    elif mutation == "prefix":
        del r["programs"]["reference_prefix"]
    elif mutation == "hlo":
        r["programs"]["reference"]["hlo_contract"]["passed"] = False
    elif mutation == "memory":
        r["programs"]["reference_prefix"]["compiled_memory"]["temp_size_in_bytes"] = (
            3 * 1024**3
        )
    else:
        del r["cases"]["tail"]
    with pytest.raises((ValueError, KeyError)):
        campaign.validate_workers(
            records, pin, layer=3, pins=pins, ledger=ledger, materialized=True
        )


@pytest.mark.parametrize(
    "mutation", (None, "dtype", "nonfinite", "count", "owner", "hash")
)
def test_original_bf16_input_capture(tmp_path, mutation):
    path = tmp_path / "tail.reference_input.npz"
    values = {"device_17": np.zeros((11, 1536), BF16).view(np.uint16)}
    if mutation == "dtype":
        values["device_17"] = values["device_17"].astype(np.float32)
    elif mutation == "nonfinite":
        values["device_17"].view(BF16)[0, 0] = np.nan
    elif mutation == "count":
        values["device_17"] = values["device_17"][:10]
    elif mutation == "owner":
        values["device_3"] = values.pop("device_17")
    np.savez_compressed(path, **values)
    digest = sha256(path.read_bytes()).hexdigest() if mutation != "hash" else "0" * 64
    if mutation is None:
        ref.verify_input_capture(path, digest, {17}, 11)
    else:
        with pytest.raises(ValueError):
            ref.verify_input_capture(path, digest, {17}, 11)


def reference_hlo(name):
    prefix = name == "reference_prefix"
    items = (
        [(4, "f32", 1)] * 2
        + [(4, "f32", 2048), (4, "f32", 576), (8, "bf16", 2048 * 640), (8, "f32", 1536)]
        if prefix
        else [(4, "f32", 4096)] * 9
        + [(4, "f32", 32), (8, "f32", 1536), (8, "f32", 256), (8, "f32", 256)]
    )
    lines = ["HloModule ref", "ENTRY %main {", "%input = bf16[1,1536] parameter(0)"]
    for i, (group, dtype, size) in enumerate(items):
        groups = (
            "{{"
            + "},{".join(
                ",".join(map(str, g)) for g in (FEATURE if group == 4 else EXPERT)
            )
            + "}}"
        )
        lines += [
            f"%p{i} = {dtype}[{size}] parameter({i+1})",
            f"%r{i} = {dtype}[{size}] all-reduce(%p{i}), replica_groups={groups}",
        ]
    names = ["greenfield_fp8_block_matmul"] * (4 if prefix else 27)
    if prefix:
        names += ["greenfield_fp8_structured_kv_b"] * 2 + [
            "greenfield_pregathered_sparse_mla"
        ]
    for i, n in enumerate(names):
        lines.append(
            f'%call{i} = bf16[1,1536] custom-call(%input), custom_call_target="tpu_custom_call", metadata={{op_name="{n}"}}'
        )
    return "\n".join(lines + ["}"])


@pytest.mark.parametrize("name", ("reference_prefix", "reference"))
def test_two_reference_graphs_have_distinct_local_inventories(name):
    hlo = reference_hlo(name)
    proof = ref.check_reference_hlo(hlo, name)
    assert proof["passed"], proof["checks"]
    assert not ref.check_reference_hlo(
        hlo.replace("all-reduce", "all-to-all", 1), name
    )["passed"]
    assert not ref.check_reference_hlo(
        hlo.replace("ENTRY %main {", "ENTRY %main {\n%bad = f32[1] infeed()"), name
    )["passed"]
    assert not ref.check_reference_hlo(
        hlo.replace(
            'custom_call_target="tpu_custom_call"',
            'custom_call_target="HostCallback"',
            1,
        ),
        name,
    )["passed"]


def test_reference_allows_retained_raw_owners_not_bf16_expansion():
    hlo = reference_hlo("reference")
    raw = hlo.replace(
        "ENTRY %main {",
        "ENTRY %main {\n%raw = u8[32,2048,1536] parameter(90)\n%copy = u8[32,2048,1536] copy(%raw)",
    )
    assert ref.check_reference_hlo(raw, "reference")["passed"]
    assert not ref.check_reference_hlo(
        raw.replace("u8[32,2048,1536]", "bf16[32,2048,1536]"), "reference"
    )["passed"]
